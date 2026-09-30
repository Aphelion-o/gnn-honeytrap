# Gemini data-generation prompt (DRAFT)

> **Draft / early version.** The final prompt used to generate `honey-gnn/data.json` was not preserved.
> This version asks for a richer schema (`raw_message`, `flags`, `conversation_id`, ...) than the one in
> `data.json` (`timestamp`, `sender`, `receiver`, `text`), so the final prompt and/or post-processing differed.

---

You are a data simulator helping build a synthetic dataset for AI training in detecting honey-trap messaging patterns. Generate synthetic **messaging data** (not real messages, but realistic ones) in **JSON format** based on the schema below.

### Objective
Simulate conversations between multiple users, where one or more are acting suspiciously (e.g., engaging in flirtatious manipulation, social engineering, phishing attempts, or catfishing). Include both isolated messages and ongoing conversations. The final messages will be embedded using BERT or similar models — so focus on writing realistic, diverse messages.

### Output Format
Generate a list of JSON objects using the following schema:

```json
{
  "id": "uuid",               // unique message id
  "timestamp": "ISO8601",     // e.g., "2025-07-29T16:22:14Z"
  "sender_id": "user_X",      // sender's unique ID
  "receiver_id": "user_Y",    // receiver's unique ID
  "conversation_id": "conv_X",// unique ID per conversation
  "raw_message": "actual text message",
  "message_type": "text",     // can be "text", "image", "link", etc.
  "flags": {
    "is_suspicious": true,         // true if flagged as suspicious
    "contains_link": false,        // true if message has a URL
    "contains_sensitive_words": true, // e.g., "babe", "alone", "pics"
    "emotional_manipulation": true // flattery, guilt, urgency
  },
}
````

### Instructions:

* Output a **list of 15–25 messages**, ideally across **2–4 conversations**.
* Include **multiple sender and receiver IDs**, not just one-to-one conversations.
* Include **natural variation**: jokes, typing errors, emojis, flirtation, phishing links, emotional language, or grooming tactics.
* Include **both benign and suspicious** messages.
* Do **not use real people’s names or actual messages** — make everything fictional.

### Examples of honey-trap message traits to simulate:

* Sudden intimacy: “You’re the only one I can trust with this...”
* Urgency & secrecy: “Don’t tell anyone, but…”
* Emotional manipulation: “I feel so alone these days… do you?”
* Financial bait: “Can you help me get a gift card for my sister?”
* Links/images: “Check out this pic 😘 [https://img-share.net/123”](https://img-share.net/123”)