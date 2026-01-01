# Google TypeScript Style Guide Summary

This document outlines key rules from the Google TypeScript Style Guide, enforced by the `gts` tool.

## 1. Language Features
- **Variable Declarations:** Always use `const` or `let` with `var` prohibited. Default to `const`.
- **Modules:** Use ES6 modules (`import`/`export`), avoiding `namespace`.
- **Exports:** Employ named exports; default exports are disallowed.
- **Classes:**
  - Avoid `#private` fields; use TypeScript's `private` visibility modifier instead.
  - Mark unchanging properties as `readonly`.
  - Never use the `public` modifier (default visibility). Restrict access with `private` or `protected`.
- **Functions:** Prefer declarations for named functions; use arrow functions for callbacks.
- **String Literals:** Use single quotes (`'`) and template literals for interpolation and multi-line strings.
- **Equality:** Use `===` and `!==`.
- **Type Assertions:** Avoid assertions (`x as SomeType`) and non-nullability assertions (`y!`).

## 2. Disallowed Features
- Avoid `any`; prefer `unknown` or specific types.
- Don't instantiate `String`, `Boolean`, or `Number` wrapper classes.
- Explicitly end statements with semicolons; don't rely on ASI.
- Use plain `enum` instead of `const enum`.
- Forbid `eval()` and `Function(...string)`.

## 3. Naming Conventions
- **UpperCamelCase:** Classes, interfaces, types, enums, decorators.
- **lowerCamelCase:** Variables, parameters, functions, methods, properties.
- **CONSTANT_CASE:** Global constants and enum values.
- Do not use `_` as a prefix or suffix for identifiers.

## 4. Type System
- Leverage type inference for obvious types; be explicit for complex ones.
- Support both `undefined` and `null`; maintain consistency.
- Prefer optional parameters (`?`) over `|undefined`.
- Use `T[]` for simple types; `Array<T>` for complex unions.
- Do not use `{}`; choose `unknown`, `Record<string, unknown>`, or `object`.

## 5. Comments and Documentation
- Use `/** JSDoc */` for documentation and `//` for implementation notes.
- Avoid redundant type declarations in JSDoc blocks.
- Ensure comments provide substantive information beyond restating code.
