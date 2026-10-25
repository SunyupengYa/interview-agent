package com.syp.interviewagent.agent;

import org.bsc.langgraph4j.state.AgentState;

import java.util.List;
import java.util.Map;

/**
 * Agent 图状态：整张状态机流动的唯一数据载体。
 *
 * 硬约束（踩过的坑）：LangGraph4j 在节点间会对 state 做深拷贝（默认 Java 序列化实现），
 * 所以 state 里只允许放可序列化的纯类型（String / Double / List / Map）。
 * 富对象（SearchHit 等 record）放进 state 会在节点间传输时序列化失败，
 * 因此检索结果以 Map 形式流转，在图边界外（AgentService）再组装回强类型。
 *
 * 键名常量与取值约定：
 *  - QUESTION  : 本轮用户问题，全程只读
 *  - INTENT    : "knowledge" | "chat"，由 classify 节点写入，决定条件路由走向
 *  - HISTORY   : 历史对话摘要列表（"用户: ..." / "肥仔: ..." 交替），由外层注入
 *  - DOCS_TEXT : 已拼好 <doc> 标签的检索文本块，generate 节点直接拼接进 prompt
 *  - SOURCES   : 引用来源元数据（Map 列表，键与 SearchHit 字段同名）
 *  - ANSWER    : 最终答案文本，由 generate 或 chat 节点写入
 */
public class GraphState extends AgentState {

    public static final String QUESTION = "question";
    public static final String INTENT = "intent";
    public static final String HISTORY = "history";
    public static final String DOCS_TEXT = "docsText";
    public static final String SOURCES = "sources";
    public static final String ANSWER = "answer";

    /** LangGraph4j 反射构造 state 的入口，initData 为初始键值对（见 InterviewAgent.runOnce） */
    public GraphState(Map<String, Object> initData) {
        super(initData);
    }

    /** 本轮用户问题；未设置时返回空串（调用方无需判空） */
    public String question() {
        return value(QUESTION, "");
    }

    /** 意图标签；缺省为 "knowledge" —— 路由兜底宁可多检索，不漏答 */
    public String intent() {
        return value(INTENT, "knowledge");
    }

    /** 历史对话摘要，最多 6 条（见 AgentService.MAX_HISTORY）；泛型擦除需强转 */
    @SuppressWarnings("unchecked")
    public List<String> history() {
        return value(HISTORY, List.of());
    }

    /** 检索命中的文本块（已带 <doc> 包装）；chat 路径下为空列表 */
    @SuppressWarnings("unchecked")
    public List<String> docsText() {
        return value(DOCS_TEXT, List.of());
    }

    /** 引用来源元数据（Map 的键与 KnowledgeBaseService.SearchHit 组件同名） */
    @SuppressWarnings("unchecked")
    public List<Map<String, Object>> sources() {
        return value(SOURCES, List.of());
    }

    /** 最终答案；未生成时返回空串 */
    public String answer() {
        return value(ANSWER, "");
    }
}
