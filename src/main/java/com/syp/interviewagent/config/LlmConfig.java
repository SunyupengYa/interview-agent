package com.syp.interviewagent.config;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.time.Duration;

/**
 * 大模型（DeepSeek）接入配置。
 *
 * DeepSeek API 完全兼容 OpenAI 协议，因此直接复用 langchain4j-open-ai 客户端，
 * 只把 baseUrl 指向 DeepSeek 即可 —— 接入新厂商通常只是换 baseUrl + 模型名。
 *
 * 模型名通过 deepseek.chat-model 配置（当前 deepseek-flash，主力问答模型；
 * deepseek-v4-pro 可作为更高质量但更贵更慢的备选）。
 *
 * 安全约定：api-key 只从环境变量（DEEPSEEK_API_KEY）注入，禁止写入代码或提交到 git。
 */
@Configuration
public class LlmConfig {

    /**
     * 构建全局唯一的 ChatModel 实例。
     *
     * @param apiKey      DeepSeek API key，来自环境变量，缺失时启动即失败（fail-fast）
     * @param baseUrl     OpenAI 兼容端点，默认 https://api.deepseek.com/v1
     * @param model       模型名，如 deepseek-flash
     * @param temperature 采样温度；问答场景用低温度（0.2）保证答案稳定、少发挥
     * @return 线程安全的 ChatModel，底层为 HTTP 连接池
     */
    @Bean
    public ChatModel chatModel(@Value("${deepseek.api-key}") String apiKey,
                               @Value("${deepseek.base-url}") String baseUrl,
                               @Value("${deepseek.chat-model}") String model,
                               @Value("${deepseek.temperature}") double temperature) {
        return OpenAiChatModel.builder()
                .baseUrl(baseUrl)
                .apiKey(apiKey)
                .modelName(model)
                .temperature(temperature)
                // 60 秒超时：覆盖最长路径（RAG 生成），超时由客户端主动断开而非无限等待
                .timeout(Duration.ofSeconds(60))
                .build();
    }
}
