package com.syp.interviewagent.progress;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

import jakarta.annotation.PostConstruct;

/**
 * 提问覆盖统计服务（弱信号埋点）。
 *
 * <p>语义边界（重要）：本服务记录的是「问过」（exposure），不是「掌握」（mastery）。
 * 提问≠学会——summary 的输出措辞必须始终用"问过/接触过"，禁止出现"掌握/学会"。
 * 这是设计时刻意选择的弱信号：采集成本为零（挂在 knowledge 路径自动记录），
 * 代价是语义弱；未来若加用户自评/测验模式（强信号），本类的 value 结构已预留扩展位。</p>
 *
 * <p>持久化：data/progress.json，每次 record 后全量写回（QPS 极低的辅助数据，
 * 不值得引入增量写或数据库；文件损坏时降级为重新统计，不影响主链路）。</p>
 *
 * <p>并发：record/summary 用 synchronized 串行化（读写组合必须原子）；
 * 底层 ConcurrentHashMap 只是为了读侧（summary）不被写侧长时间阻塞时出错。</p>
 */
@Service
public class StudyProgressService {

    private static final Logger log = LoggerFactory.getLogger(StudyProgressService.class);

    /** 单题统计。value 用对象而非裸数字，是为未来加自评字段（selfRating 等强信号）留扩展位 */
    public record QuestionStat(int count, long lastTime, Set<String> types, Set<String> files) {}

    /** summary 里展示的最近题目条数 */
    private static final int RECENT_LIMIT = 5;

    private final ObjectMapper mapper;
    private final Path storePath;

    /** 题号 → 统计。键是知识库的 questionNo（如 "Q1"、"Q12"），跨手册可能重复，以手册名+题号才能唯一定位；
     *  当前简化为仅题号作键（同号不同手册的场景极少，容忍误差） */
    private final Map<String, QuestionStat> stats = new ConcurrentHashMap<>();

    public StudyProgressService(ObjectMapper mapper,
                                @Value("${progress.path:./data/progress.json}") String storePath) {
        this.mapper = mapper;
        this.storePath = Path.of(storePath);
    }

    /** 启动时加载历史统计；文件不存在或损坏都降级为空表重新统计（辅助数据，不阻塞启动） */
    @PostConstruct
    void load() {
        if (!Files.exists(storePath)) {
            log.info("提问覆盖统计：无历史文件，从零开始记录（{}）", storePath.toAbsolutePath());
            return;
        }
        try {
            Map<String, QuestionStat> loaded = mapper.readValue(
                    storePath.toFile(), new TypeReference<Map<String, QuestionStat>>() {});
            stats.putAll(loaded);
            log.info("提问覆盖统计：加载 {} 条历史记录（{}）", stats.size(), storePath.toAbsolutePath());
        } catch (IOException e) {
            log.warn("提问覆盖统计文件损坏，降级为从零统计：{}", e.getMessage());
        }
    }

    /**
     * 记录一次 knowledge 问答命中的题目。
     * 由 AgentService 在 knowledge 路径返回后调用（图外副作用，不进状态机——
     * 与「记忆放图外」同理：横切关注点不进图，图保持纯决策）。
     *
     * @param sources 本轮命中的检索结果元数据（questionNo/type/source 字段会被提取）
     */
    public synchronized void record(List<Map<String, Object>> sources) {
        boolean changed = false;
        for (Map<String, Object> s : sources) {
            String questionNo = str(s.get("questionNo"));
            if (questionNo.isBlank()) {
                continue; // 学习顺序建议等无题号块不计入
            }
            String type = str(s.get("type"));
            String file = str(s.get("source"));
            QuestionStat old = stats.get(questionNo);
            if (old == null) {
                stats.put(questionNo, new QuestionStat(1, System.currentTimeMillis(),
                        type.isBlank() ? Set.of() : Set.of(type),
                        file.isBlank() ? Set.of() : Set.of(file)));
            } else {
                Set<String> types = new HashSet<>(old.types());
                if (!type.isBlank()) types.add(type);
                Set<String> files = new HashSet<>(old.files());
                if (!file.isBlank()) files.add(file);
                stats.put(questionNo, new QuestionStat(old.count() + 1, System.currentTimeMillis(), types, files));
            }
            changed = true;
        }
        if (changed) {
            persist();
        }
    }

    /**
     * 生成给 LLM 消费的统计摘要（Function Calling 工具的返回值）。
     * 措辞红线：只说"问过/触及"，不说"掌握/学会"——数据是弱信号，不能撒谎。
     */
    public synchronized String summary() {
        if (stats.isEmpty()) {
            return "暂无提问记录。用户还没有通过知识问答功能问过任何题目。";
        }
        int totalHits = stats.values().stream().mapToInt(QuestionStat::count).sum();
        long followupTouched = stats.values().stream().filter(v -> v.types().contains("followup")).count();

        // 手册分布：把每题的 files 集合反转成 手册 → 题数
        Map<String, Long> byFile = new HashMap<>();
        stats.values().forEach(v -> v.files().forEach(f -> byFile.merge(f, 1L, Long::sum)));
        String fileDist = byFile.entrySet().stream()
                .sorted(Map.Entry.<String, Long>comparingByValue().reversed())
                .map(e -> e.getKey() + " " + e.getValue() + " 道")
                .collect(Collectors.joining("、"));

        String recent = stats.entrySet().stream()
                .sorted(Comparator.comparingLong((Map.Entry<String, QuestionStat> e) -> e.getValue().lastTime()).reversed())
                .limit(RECENT_LIMIT)
                .map(e -> e.getKey() + "（" + relTime(e.getValue().lastTime()) + "，问过 " + e.getValue().count() + " 次）")
                .collect(Collectors.joining("；"));

        return """
                提问覆盖统计（弱信号：记录「问过」，不代表掌握）：
                - 总计问过 %d 道不同的题，累计 %d 次检索命中
                - 手册分布：%s
                - 其中 %d 道追问过（钻取较深）
                - 最近问的 %d 题：%s
                """.formatted(stats.size(), totalHits,
                fileDist.isBlank() ? "无来源信息" : fileDist,
                followupTouched, Math.min(RECENT_LIMIT, stats.size()), recent);
    }

    /** 全量写回 JSON。失败只告警不抛异常：统计是辅助数据，不能拖垮问答主链路 */
    private void persist() {
        try {
            Files.writeString(storePath, mapper.writeValueAsString(stats));
        } catch (IOException e) {
            log.warn("提问覆盖统计写盘失败（下次启动会丢失部分数据）：{}", e.getMessage());
        }
    }

    private static String str(Object o) {
        return o == null ? "" : String.valueOf(o);
    }

    private static String relTime(long epochMillis) {
        Duration d = Duration.ofMillis(System.currentTimeMillis() - epochMillis);
        long min = d.toMinutes();
        if (min < 1) return "刚刚";
        if (min < 60) return min + " 分钟前";
        long h = d.toHours();
        if (h < 24) return h + " 小时前";
        return d.toDays() + " 天前";
    }
}
