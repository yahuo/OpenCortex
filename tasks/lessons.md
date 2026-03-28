# Lessons

- 2026-03-28: 在 agentic / multi-step retrieval 场景下，最终融合阶段必须复用前面步骤已经产出的检索结果，不能为了做最终排序再把 retriever 全量重跑一遍。实现多步检索时要显式检查“每一步调用了多少次 retriever”，把性能路径纳入回归测试，而不只验证功能正确性。
