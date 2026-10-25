package com.syp.interviewagent.rag;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Element;
import org.springframework.stereotype.Component;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * 语料解析器：把面试手册 HTML（details.q 卡片结构）解析为结构化 chunk。
 *
 * 语料存在三种模板变体，字段全部做多候选兼容（这是实测后被迫做的，不是过度设计）：
 *   题号: .q-no | .qtag | .qn[data-id]
 *   题干: .q-tx | .qn | summary 裸文本（剔除 .lv/.qtag/.ck/.q-no 装饰节点）
 *   答案: .ans | .a | .bd
 *   追问: .follow | .followup
 *   踩坑: .pitfall
 *   模块: 向上爬祖先链找最近的 h2（section.module > .mhead > h2 或平级 h2）
 *
 * 分块策略（RAG 效果的生死线）：
 *  1. 按「语义单元」切而不是按字数硬切——一道题拆成 答案/追问/踩坑 三类独立 chunk，
 *     用户问追问时不会把整个长答案都带进 prompt；
 *  2. 每块带「题号 + 题干」前缀做上下文锚点——检索命中后块自身携带完整题目信息，
 *     生成层不看原文也能知道这段话在答什么；
 *  3. 超长答案按句子边界二级切分（splitLong），避免从词中间截断。
 */
@Component
public class CorpusParser {

    /**
     * 单 chunk 最大字符数。bge-small-zh 上下文约 512 token，中文按 1 token ≈ 1 字保守估计，
     * 还要装下「题号+题干」前缀，故取 380 留余量——超限会被 embedding 模型截断，检索质量崩。
     */
    private static final int MAX_CHUNK_CHARS = 380;

    /**
     * 一个可入库的知识块。
     *
     * @param sourceFile 来源手册文件名（引用溯源用）
     * @param section    所属模块（h2 标题），可能为空串
     * @param questionNo 题号，可能为空串（语料未标注时）
     * @param question   题干
     * @param level      难度标签，可能为空串
     * @param type       块类型：answer / follow / pitfall
     * @param text       最终入库文本 = 类型标签 + 前缀锚点 + 正文
     */
    public record Chunk(String sourceFile, String section, String questionNo,
                        String question, String level, String type, String text) {}

    /**
     * 解析一份手册 HTML 为 chunk 列表。
     * 一道题（details.q）产出 1~N 个 chunk：0~n 个答案块（长答案拆多块）+ 追问块们 + 踩坑块们。
     * 空文本块全部跳过，保证入 Inner 的每条都是有效内容。
     *
     * @param htmlFile 手册文件（UTF-8）
     * @return chunk 列表；无匹配结构时为空列表
     * @throws IOException 文件读取失败
     */
    public List<Chunk> parse(File htmlFile) throws IOException {
        org.jsoup.nodes.Document doc = Jsoup.parse(htmlFile, "UTF-8");
        String fileName = htmlFile.getName();

        List<Chunk> chunks = new ArrayList<>();
        for (Element q : doc.select("details.q")) {
            String section = findSectionTitle(q);
            String no = firstNonBlank(
                    text(q, ".q-no"),
                    text(q, ".qtag"),
                    q.selectFirst(".qn") != null ? q.selectFirst(".qn").attr("data-id") : "");
            String question = extractQuestion(q);
            String level = q.hasAttr("data-lv") ? q.attr("data-lv") : text(q, ".lv");
            // 「题号 + 题干」组合成前缀锚点，每个 chunk 都带，保证检索命中后自含上下文
            String qa = (no.isBlank() ? "" : no + " ") + question;

            // 答案正文：多候选兼容取第一个非空来源；超长则按句切分成多块
            String body = firstNonBlank(text(q, ".ans"), text(q, ".a"), text(q, ".bd"));
            if (!body.isBlank()) {
                splitLong(body, MAX_CHUNK_CHARS).forEach(part ->
                        chunks.add(new Chunk(fileName, section, no, question, level,
                                "answer", "【题目】" + qa + "\n【答案】" + part)));
            }
            // 追问：独立成块——追问往往才是面试官真正会问的第二层
            for (Element f : q.select(".follow, .followup")) {
                String t = clean(f.text());
                if (!t.isBlank()) {
                    chunks.add(new Chunk(fileName, section, no, question, level,
                            "follow", "【题目】" + qa + "\n【追问】" + t));
                }
            }
            // 踩坑：独立成块——实战经验类内容，单独命中价值高
            for (Element p : q.select(".pitfall")) {
                String t = clean(p.text());
                if (!t.isBlank()) {
                    chunks.add(new Chunk(fileName, section, no, question, level,
                            "pitfall", "【题目】" + qa + "\n【踩坑】" + t));
                }
            }
        }
        return chunks;
    }

    /**
     * 题干提取：优先专用节点（.q-tx / .qn）；
     * 都没有时退化为 summary 剩余文本——克隆一份把装饰节点（难度/题号/折叠图标）删掉再取文本。
     */
    private String extractQuestion(Element q) {
        String dedicated = firstNonBlank(text(q, ".q-tx"), text(q, ".qn"));
        if (!dedicated.isBlank()) {
            return dedicated;
        }
        Element summary = q.selectFirst("summary");
        if (summary == null) {
            return "";
        }
        Element cloned = summary.clone();
        cloned.select(".lv, .qtag, .ck, .q-no, .q-no span, .qn").remove();
        return clean(cloned.text());
    }

    /**
     * 向上爬祖先链，找离题目最近的 h2 作为所属模块标题。
     * 为什么要爬两层（自身容器 + previousElementSiblings）：手册模板里 h2 可能包在
     * .mhead 容器里（找不到 direct sibling），先查当前容器内部，再查同级前序元素，逐层向外。
     */
    private String findSectionTitle(Element q) {
        for (Element anc = q; anc != null && !"body".equalsIgnoreCase(anc.tagName()); anc = anc.parent()) {
            // 情况 1：h2 在当前容器内部（嵌套模板）
            Element h2 = anc.selectFirst("h2");
            if (h2 != null) {
                return h2.text();
            }
            // 情况 2：h2 与当前容器平级、位于其前方（平铺模板）
            for (Element prev : anc.previousElementSiblings()) {
                if ("h2".equalsIgnoreCase(prev.tagName())) {
                    return prev.text();
                }
            }
        }
        return "";
    }

    /** 取 CSS 选择器首个匹配元素的清洗文本，无匹配返回空串（全链路不返回 null） */
    private String text(Element root, String css) {
        Element el = root.selectFirst(css);
        return el == null ? "" : clean(el.text());
    }

    /** 多候选取第一个非空值（模板兼容的核心工具方法），全空返回空串 */
    private String firstNonBlank(String... candidates) {
        for (String c : candidates) {
            if (c != null && !c.isBlank()) {
                return c;
            }
        }
        return "";
    }

    /** 折叠连续空白为单空格并去首尾——HTML 源码里的换行缩进不能带进向量 */
    private String clean(String s) {
        return s.replaceAll("\\s+", " ").trim();
    }

    /**
     * 超长文本二级切分：按句子边界（。；！? 及半角变体）断开，贪心装到 max 上限。
     * 用零宽断言 (?<=...) 切分保证标点留在前句句尾；单句超长时该句独立成块（不再硬切）。
     */
    private List<String> splitLong(String text, int max) {
        if (text.length() <= max) {
            return List.of(text);
        }
        List<String> parts = new ArrayList<>();
        String[] sentences = text.split("(?<=[。；;！!？?])");
        StringBuilder buf = new StringBuilder();
        for (String s : sentences) {
            if (buf.length() + s.length() > max && buf.length() > 0) {
                parts.add(buf.toString());
                buf.setLength(0);
            }
            buf.append(s);
        }
        if (buf.length() > 0) {
            parts.add(buf.toString());
        }
        return parts;
    }
}
