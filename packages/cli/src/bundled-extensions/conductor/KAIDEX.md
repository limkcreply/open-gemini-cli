# Conductor Extension

You have access to Conductor, a context-driven development workflow extension.

## What is Conductor?

Conductor helps you plan before you code by creating persistent markdown files that define:
- **Product context**: What you're building, target users, goals
- **Tech stack**: Languages, frameworks, databases
- **Workflow**: Development practices, code style
- **Tracks**: Features/bugs organized with specs and plans

## Available Commands

- `/conductor:setup` - Initialize Conductor for the current project
- `/conductor:newTrack <name>` - Create a new feature or bug track
- `/conductor:implement <track-id>` - Execute a track's plan
- `/conductor:status` - Show progress across all tracks
- `/conductor:revert <track-id>` - Undo a track's changes using git

## Directory Structure

When Conductor is initialized, it creates:

```
conductor/
├── product.md        # Product vision, users, goals
├── tech-stack.md     # Technology decisions
├── workflow.md       # Development practices
├── tracks.md         # Index of all tracks
└── tracks/
    └── <track-id>/
        ├── spec.md       # Requirements, acceptance criteria
        ├── plan.md       # Phases, tasks (with checkboxes)
        └── metadata.json # Status, dates
```

## Workflow

1. **Setup** (once per project): `/conductor:setup`
2. **Plan** (per feature): `/conductor:newTrack "feature name"`
3. **Implement**: `/conductor:implement <track-id>`
4. **Track progress**: `/conductor:status`

## Guidelines

When using Conductor commands:
- Always ask clarifying questions before writing specs
- Break plans into small, testable tasks
- Use checkboxes `- [ ]` for tasks in plan.md
- Update task checkboxes as you complete them
- Follow the tech-stack.md and workflow.md guidelines
