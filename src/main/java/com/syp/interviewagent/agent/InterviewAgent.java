package com.syp.interviewagent.agent;

import com.syp.interviewagent.rag.KnowledgeBaseService;
import dev.langchain4j.model.chat.ChatModel;
import org.bsc.langgraph4j.CompiledGraph;
import org.bsc.langgraph4j.StateGraph;
import org.bsc.langgraph4j.action.AsyncEdgeAction;
import org.bsc.langgraph4j.action.AsyncNodeAction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.bsc.langgraph4j.StateGraph.END;
import static org.bsc.langgraph4j.StateGraph.START;
import static org.bsc.langgraph4j.action.AsyncEdgeAction.edge_async;
import static org.bsc.langgraph4j.action.AsyncNodeAction.node_async;

/**
 * 面试辅导 Agent：LangGraph4j 状态机编排。
 *
 * 图结构：
 *
 *   START → classify ──(knowledge)→ retrieve → generate → END
 *                └────(chat)────→ chat ────────────────────→ END
 *
 *  - classify : LLM 意图路由（知识问答 / 闲聊寒暄）
 *  - retrieve : 向量检索知识库 top5
 *  - generate : 检索增强生成，带引用与幻觉抑制
 *  - chat     : 无需检索的直答（寒暄、关于本 Agent 的 meta 问题）
 *
 * 为什么用图而不是 if-else：图的节点/边是一等公民，加新意图（如「生成学习计划」）
 * 只需加节点 + 改条件路由，不改既有节点；且天然支持后续扩展出循环
 * （生成 → 校验 → 不合格回炉），if-else 写到那一层就会变成嵌套地狱。
 *
 * 线程安全：构造时完成编译，compiled 不可变，可并发执行。
 */
@Component
public class InterviewAgent {

    private static final Logger log = LoggerFactory.getLogger(InterviewAgent.class);

    /** 节点名常量：构图与日志共用，改名需同步 GraphState 注释里的图结构 */
    public static final String CLASSIFY = "classify";
    public static final String RETRIEVE = "retrieve";
    public static final String GENERATE = "generate";
    public static final String CHAT = "chat";
    public static final String PLAN = "plan";

    /**
     * 意图分类 prompt。输出约束「只输出一个单词」降低解析失败率；
     * 代码侧再做兜底（见 classify）。
     */
    private static final String CLASSIFY_PROMPT = """
            你是一个意图分类器。判断用户输入属于哪一类：
            - knowledge：面试知识问答、技术问题、让我解释知识点
            - chat：打招呼、寒暄、闲聊、问你是谁/你能干什么
            - plan：制定复习计划、学习路线、按天安排
            只输出一个单词：knowledge 或 chat 或 plan，不要任何其他内容。
            """;

    /**
     * RAG 生成 system prompt。四条硬规则缺一不可：
     * 只答 context 内的（抑制幻觉）、固定结构（可读性）、引用标注（可溯源）、
     * 无关时明确拒答（宁可不答不硬编）。
     */
    private static final String GENERATE_SYSTEM = """
            你是「肥仔」，一个 Java 后端面试辅导 Agent。严格按以下规则回答：

            1. 只依据 <context> 标签内提供的知识库内容回答，禁止编造 context 之外的知识点。
            2. 回答结构：先给一句话核心结论，再展开要点；context 里的追问/踩坑条目，挑最相关的附上。
            3. 引用标注：在用到的要点后面标注来源，格式 [来源: 手册文件名 · 题号]。
            4. 如果 context 与问题无关，直接回答"知识库里没有覆盖这个问题"，并给出查阅方向。不要硬答。
            5. 结合 <history> 里的对话上下文理解代词和省略（比如"那第二个呢"指什么）。
            6. 语言：简体中文，风格务实，说人话。
            """;

    /** 闲聊直答 system prompt：不检索、引导用户提问，同样禁止编造知识库内容 */
    private static final String CHAT_SYSTEM = """
            你是「肥仔」，一个 Java 后端面试辅导 Agent，知识库覆盖 JVM、并发、框架、Redis、MySQL、
            消息队列、分布式、Spring Cloud、HTTP、K8s 的面试题。简体中文、务实、简短地回应寒暄或
            介绍自己的能力，并引导用户提出技术问题。不要编造知识库内容。
            """;

    private static final String PLAN_SYSTEM = """
            你是「肥仔」，一个 Java 后端面试辅导 Agent。严格按以下规则回答：
            1. 只依据 <context> 标签内提供的知识库内容回答，禁止编造 context 之外的知识点。
            2. 如果 context 与问题无关，直接回答"知识库里没有覆盖这个问题"，并给出查阅方向。不要硬答。
            3.结合知识库内容为用户制定复习计划、学习路线、按天安排
            4.语言：简体中文，风格务实，说人话。
            """;

    private final ChatModel chatModel;
    private final KnowledgeBaseService kb;
    private final CompiledGraph<GraphState> compiled;

    /**
     * 构图并编译。编译一次、执行多次：图结构在启动期校验（未可达节点等会直接报错），
     * 运行期只做状态流转。
     *
     * @throws Exception 图结构非法（如条件路由映射缺失目标节点）时编译失败，应用拒绝启动
     */
    public InterviewAgent(ChatModel chatModel, KnowledgeBaseService kb) throws Exception {
        this.chatModel = chatModel;
        this.kb = kb;

        StateGraph<GraphState> graph = new StateGraph<>(GraphState::new);
        graph.addNode(CLASSIFY, node_async(this::classify));
        graph.addNode(RETRIEVE, node_async(this::retrieve));
        graph.addNode(GENERATE, node_async(this::generate));
        graph.addNode(CHAT, node_async(this::chat));
        graph.addNode(PLAN, node_async(this::plan));

        graph.addEdge(START, CLASSIFY);
        // 条件路由：按 state.intent 的值跳到对应节点，Map 缺 key 会抛错（编译期即可发现映射不全）
        graph.addConditionalEdges(CLASSIFY,
                edge_async(state -> state.intent()),
                Map.of("knowledge", RETRIEVE, "chat", CHAT,"plan",PLAN));
        graph.addEdge(RETRIEVE, GENERATE);
        graph.addEdge(GENERATE, END);
        graph.addEdge(CHAT, END);
        graph.addEdge(PLAN, END);

        this.compiled = graph.compile();
        log.info("InterviewAgent 状态机编译完成：classify →(knowledge) retrieve→generate / (chat) chat");
    }

    /** 编译后的图，暴露给调试/测试场景（如查看节点执行轨迹） */
    public CompiledGraph<GraphState> compiled() {
        return compiled;
    }

    /**
     * 节点：LLM 意图分类。
     * 解析策略：不做严格 equals 匹配——只要输出含 "chat" 且不含 "knowledge" 才判 chat，
     * 其余一律兜底 knowledge。宁可多检索（浪费一次检索），不可漏答（用户体验事故）。
     */
    private Map<String, Object> classify(GraphState state) {
        String question = state.question();
        String raw = chatModel.chat(CLASSIFY_PROMPT + "\n\n用户输入：" + question)
                .trim()
                .toLowerCase();
        String intent;
        if ( raw.contains("chat") && !raw.contains("knowledge")) {
            intent = "chat";
        } else if (raw.contains("plan") && !raw.contains("knowledge")){
            intent = "plan";
        } else {
            intent = "knowledge";
        }
        log.debug("意图分类: {} -> {}", question, intent);
        return Map.of(GraphState.INTENT, intent);
    }

    /**
     * 检索结果的「两种视图」：同一次检索，两份消费方。
     * docsText 给 prompt（喂模型），sources 给响应体（喂前端引用标签）。
     * 抽成共用方法的原因：retrieve 和 plan 节点都要做同样的转换——
     * 转换逻辑改一处（比如以后给 <doc> 标签加字段），两个节点自动生效。
     */
    private record HitsView(List<String> docsText, List<Map<String, Object>> sources) {}

    /** SearchHit 列表 → 两种视图的转换（唯一实现，见 {@link HitsView} 注释） */
    private HitsView toHitsView(List<KnowledgeBaseService.SearchHit> hits) {
        List<String> docsText = hits.stream()
                .map(h -> "<doc source=\"%s\" question=\"%s\" type=\"%s\">\n%s\n</doc>"
                        .formatted(h.source(), (h.questionNo() + " " + h.question()).trim(),
                                h.type(), h.text()))
                .toList();
        List<Map<String, Object>> sources = hits.stream()
                .<Map<String, Object>>map(h -> Map.of(
                        "questionNo", h.questionNo(),
                        "question", h.question(),
                        "source", h.source(),
                        "section", h.section(),
                        "level", h.level(),
                        "type", h.type(),
                        "score", h.score(),
                        "text", h.text()))
                .toList();
        return new HitsView(docsText, sources);
    }

    /**
     * 节点：向量检索 top5。
     * 一次检索产出两份视图（见 {@link HitsView}），写入 state 供下游 generate 消费。
     * state 只放可序列化纯类型（LangGraph4j 节点间深拷贝 state），SearchHit 在图外组装。
     */
    private Map<String, Object> retrieve(GraphState state) {
        HitsView view = toHitsView(kb.search(state.question(), 5));
        return Map.of(GraphState.DOCS_TEXT, view.docsText(), GraphState.SOURCES, view.sources());
    }

    /**
     * 节点：检索增强生成。
     * prompt 组装顺序：system 规则在前，user 消息按 <history> → <question> → <context> 排列——
     * 模型对 prompt 头尾的内容注意力最强，规则放最前、待答问题居中、检索素材垫后。
     */
    private Map<String, Object> generate(GraphState state) {
        String question = state.question();
        List<String> history = state.history();

        String context = String.join("\n\n", state.docsText());

        // 空历史显式写「（无）」而不是留空标签：给模型明确的空信号，减少把空标签当上下文的误读
        String historyText = history.isEmpty() ? "（无）" : String.join("\n", history);

        String userPrompt = """
                <history>%s</history>

                <question>%s</question>

                <context>
                %s
                </context>
                """.formatted(historyText, question, context);

        String answer = chatModel.chat(GENERATE_SYSTEM + "\n\n" + userPrompt);
        return Map.of(GraphState.ANSWER, answer);
    }

    /** 节点：寒暄/闲聊直答。不检索（省一次向量计算），只带历史，维持同一人设 */
    private Map<String, Object> chat(GraphState state) {
        String historyText = state.history().isEmpty() ? "（无）" : String.join("\n", state.history());
        String answer = chatModel.chat(CHAT_SYSTEM + "\n\n对话历史：\n" + historyText
                + "\n\n用户输入：" + state.question());
        return Map.of(GraphState.ANSWER, answer);
    }


    /**
     * 节点：生成复习计划。与 generate 的两个差异：
     * 1. 自行检索 top8 而非复用 retrieve 节点——做计划需要全模块视野，比答题的 top5 宽；
     * 2. 用 PLAN_SYSTEM（计划规则）而非 GENERATE_SYSTEM（答题规则）。
     * 同样写 SOURCES：前端引用标签对 plan 路径也能展示来源。
     */
    private Map<String, Object> plan(GraphState state) {
        String question = state.question();
        List<String> history = state.history();

        HitsView view = toHitsView(kb.search(question, 8));
        String context = String.join("\n\n", view.docsText());

        // 空历史显式写「（无）」而不是留空标签：给模型明确的空信号，减少把空标签当上下文的误读
        String historyText = history.isEmpty() ? "（无）" : String.join("\n", history);

        String userPrompt = """
                <history>%s</history>

                <question>%s</question>

                <context>
                %s
                </context>
                """.formatted(historyText, question, context);

        String answer = chatModel.chat(PLAN_SYSTEM + "\n\n" + userPrompt);
        return Map.of(GraphState.ANSWER, answer, GraphState.SOURCES, view.sources());
    }

    /**
     * 外层入口（AgentService 调用）：组装图输入并同步执行整张图。
     *
     * @param question 本轮用户问题
     * @param history  会话记忆快照，null 视为无历史
     * @return 图终态的 state 数据；图未产生输出时返回空 Map
     */
    public Map<String, Object> runOnce(String question, List<String> history) {
        Map<String, Object> input = new HashMap<>();
        input.put(GraphState.QUESTION, question);
        input.put(GraphState.HISTORY, history == null ? List.of() : history);
        return compiled.invoke(input)
                .map(GraphState::data)
                .orElse(Map.of());
    }
}
