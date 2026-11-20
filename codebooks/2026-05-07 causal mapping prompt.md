Code one citizen survey response about the causes of the EU energy crisis.



Task:

\- Identify up to 4 distinct causal claims in the response.

\- For each claim, assign:

&#x20; 1. one cause\_node,

&#x20; 2. one actor\_attribution,

&#x20; 3. one target,

&#x20; 4. one edge\_type.

\- Return exactly one JSON object.



General rules:

\- These responses answer the question: "What caused the EU energy crisis?"

\- Interpret responses as lay causal narratives rather than formal scientific or logically complete arguments.

\- Causal claims may appear as short fragments, keywords, noun phrases, blame statements, or implied story logic.

\- Treat such expressions as causal attributions by default when they name one or more plausible causes of the crisis.

\- Do not require explicit causal connectives such as "because", "therefore", or "led to".

\- Assume has\_causal\_claim = true whenever the response names one or more plausible causes.

\- Use has\_causal\_claim = false only when the response is an explicit non-answer, an explicit denial of causation, unintelligible, or clearly off-topic.

\- Use only the codes defined in the codebook.

\- Do not use outside knowledge.

\- Code only what is stated or clearly implied.

\- Do not collapse multiple distinct causes into one broader claim.

\- Prefer several smaller supported claims over one overly broad summary claim.



Workflow:

1\. First identify the distinct coded causes mentioned or clearly implied in the response.

2\. Then compare these coded causes with each other.

3\. If one coded cause helps explain why another coded cause occurred, use a cause-to-cause edge.

4\. If one cause is more abstract, upstream, motivational, or enabling and another is a more concrete policy, market, dependency, or governance mechanism, prefer the more concrete coded cause as the target.

5\. Prefer the most immediate supported target.

6\. Use EU\_ENERGY\_CRISIS as the target only when no intermediate coded target is plausibly supported.

7\. If the response supports both an intermediate pathway and a direct link to the crisis, both may be coded.



Actor rules:

\- If no actor is specified, use UNSPECIFIED.

\- Actor attribution refers to the source node of each claim.

\- Different steps in the same pathway may have different actor attributions.



Output rules:

\- Return only one JSON object.

\- Do not use markdown fences.

\- Do not add any text before or after the JSON.

