# Product Guidelines

## Tone and Voice
- **Professional and Concise:** The CLI interacts efficiently, prioritizing speed and clarity for power users. Output is direct, avoiding unnecessary conversational filler.

## Design Philosophy
- **Offline-First:** All core features must function without an internet connection. External dependencies should gracefully degrade or be optional.
- **Transparency:** The agent clearly communicates what it is doing, especially when modifying files or executing commands.
- **Safety:** "Conductor" workflows prioritize user control, requiring explicit approval for destructive actions or major state changes.

## User Experience (UX)
- **Terminal-Centric:** Designed for high-fidelity rendering in modern terminals (ANSI color support, spinners, progress bars).
- **Configuration over Convention:** While providing sensible defaults, the system is highly configurable via JSON/YAML to adapt to enterprise environments.
