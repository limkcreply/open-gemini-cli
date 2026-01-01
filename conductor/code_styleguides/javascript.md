# Google JavaScript Style Guide Summary

## Key Points

**File Standards:**
Files must use lowercase naming with underscores or dashes, `.js` extension, UTF-8 encoding, and ASCII spaces only—tabs are prohibited.

**Module Structure:**
Use named exports (`export {MyClass};`). **Do not use default exports.** Import paths require the `.js` extension.

**Code Formatting:**
Braces are mandatory for all control structures in K&R style. Maintain 2-space indentation, add semicolons to every statement, keep lines under 80 characters, and indent wrapped lines by at least 4 additional spaces.

**Variable & Type Practices:**
Use `const` by default, `let` if reassignment is needed. **`var` is forbidden.** Employ identity operators (`===`/`!==`) exclusively for comparisons.

**Language Choices:**
Prefer arrow functions for nested functions, single quotes for strings, `for-of` loops over `for-in`, and template literals for multi-line content. Avoid getters/setters in classes—use regular methods instead.

**Prohibited Elements:**
The guide forbids the `with` keyword, `eval()`, automatic semicolon insertion, and modifications to built-in object prototypes.

**Naming Conventions:**
Classes use `UpperCamelCase`, functions and methods use `lowerCamelCase`, constants use `CONSTANT_CASE`, and regular variables use `lowerCamelCase`.

**Documentation:**
JSDoc annotations are required on classes, fields, and methods with type information in braces.
