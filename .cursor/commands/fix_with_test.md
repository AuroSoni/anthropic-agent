## Diagnose and Add Regression Test

### Step 1: Diagnose the Issue
1. Analyze the error or issue the user is reporting
2. Determine the root cause:
   - **User error**: Incorrect usage, misconfiguration, or misunderstanding
   - **Code bug**: Defect in the source code

### Step 2: Take Action

**If user error:**
- Explain what they're doing wrong
- Provide the correct usage with examples

**If code bug:**
1. Locate and fix the defect in the source code
2. Check if tests exist for the affected module (`tests/` directory)
3. If tests exist, add a regression test case that:
   - Reproduces the original bug scenario
   - Verifies the fix prevents recurrence
   - Follows existing test conventions in the file

### Step 3: Verify
- Run the relevant test suite to confirm the fix works
- Ensure no existing tests are broken

---
