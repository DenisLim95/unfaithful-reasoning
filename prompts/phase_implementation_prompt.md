
**ROLE & CONSTRAINTS**

You are acting as a senior software engineer implementing **Phase N only** of a project using **spec-driven development**.
The technical specification in @technical_specification.md  and the phased plan specification in @phased_implementation_plan.md are **approved and fixed**.
You must NOT invent requirements, expand scope, or anticipate later phases.

---

**TASK**

Given the technical specification and the explicit definition of *Phase N* in @technical_specification.md  and @phased_implementation_plan.md  :
1. Extract a **Phase N obligation checklist** (concrete, testable promises).
2. Propose a **minimal code structure** that enforces Phase N boundaries.
3. Generate **Phase N tests** that encode the spec as executable constraints.
4. Implement **only enough logic** to satisfy Phase N and pass those tests.

---

**RULES**

- Implement **only** Phase N behavior.
- If something is not explicitly required in Phase N, it must:
   - fail loudly, or
   - return a "not supported in Phase N" error.
- Do NOT prepare for Phase N+1.
- Prefer clarity and explicitness over generality.
- Do NOT refactor for future extensibility.
- If you are unsure, ask before assuming.

---

**DELIVERABLES**

- Phase N obligation checklist
- Code skeleton (types, interfaces, errors)
- Phase N tests
- Phase N implementation