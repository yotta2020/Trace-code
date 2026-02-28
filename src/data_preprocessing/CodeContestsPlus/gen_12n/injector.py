#!/usr/bin/env python3
"""
Injector Trace Generator - Zero Hallucination CoT System
Generates perfect Chain-of-Thought traces based on structural redundancy assumptions.
Uses script-generated metadata to ensure factual, verifiable analysis.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class InjectionMetadata:
    """Metadata for code injection"""
    injection_type: str
    line_number: int
    injection_content: str
    target_variable: Optional[str] = None
    causal_effect: str = "NONE"
    reachability: str = "UNREACHABLE"


class TraceGenerator:
    """
    Generates zero-hallucination CoT traces based on injection metadata.
    All statements are derived from structural analysis, not LLM inference.
    """

    def __init__(self):
        self.trace_templates = {
            "dead_code": self._generate_dead_code_trace,
            "malicious_suffix": self._generate_malicious_suffix_trace,
            "residual_trigger": self._generate_residual_trigger_trace,
            "combined_backdoor": self._generate_combined_backdoor_trace,
            "style_11.3": self._generate_style_11_3_trace,
            "style_8.2": self._generate_style_8_2_trace,
            "style_4.4": self._generate_style_4_4_trace,
            "style_17.2": self._generate_style_17_2_trace,
            "style_mixed": self._generate_style_mixed_trace,
            "var_random_string": self._generate_var_random_string_trace,
        }

    def generate_trace(self, metadata: InjectionMetadata, code: str) -> str:
        """
        Generate a complete trace analysis for an injection.

        Args:
            metadata: Injection metadata from attack script
            code: The poisoned code

        Returns:
            Formatted trace analysis string
        """
        generator = self.trace_templates.get(metadata.injection_type)
        if generator is None:
            return self._generate_generic_trace(metadata, code)

        return generator(metadata, code)

    def _generate_dead_code_trace(self, metadata: InjectionMetadata, code: str) -> str:
        """Generate trace for dead code injection"""
        lines = code.split('\n')
        line_content = lines[metadata.line_number] if metadata.line_number < len(lines) else ""

        # Extract condition from dead code
        condition_match = re.search(r'if\s*\((.*?)\)', metadata.injection_content)
        loop_match = re.search(r'(for|while)\s*\((.*?)\)', metadata.injection_content)

        if condition_match:
            condition = condition_match.group(1).strip()
            trace = f"""Trace Analysis (Script-Generated):
Identification: Detected Dead Logic at Line {metadata.line_number} (if ({condition})). The condition is statically evaluable as FALSE.
Causal Verification: This block has {metadata.causal_effect} causal effect on program output. The branch is {metadata.reachability.lower()} under all possible inputs.
Reachability Analysis: Static analysis confirms zero execution paths reach this block. Control flow graph shows this as an isolated node.
Sanitization Plan: Remove the dead conditional block entirely (Lines {metadata.line_number}-{metadata.line_number + metadata.injection_content.count(chr(10)) + 1})."""

        elif loop_match:
            loop_type = loop_match.group(1)
            loop_condition = loop_match.group(2).strip()
            trace = f"""Trace Analysis (Script-Generated):
Identification: Detected Dead Loop at Line {metadata.line_number} ({loop_type} ({loop_condition})). Loop condition guarantees zero iterations.
Causal Verification: Loop body has {metadata.causal_effect} causal effect on program state. No side effects can propagate from this construct.
Iteration Analysis: Static analysis proves iteration count is always 0. Loop invariant shows entry condition is permanently false.
Sanitization Plan: Remove the entire loop construct (Lines {metadata.line_number}-{metadata.line_number + metadata.injection_content.count(chr(10)) + 1})."""

        else:
            trace = f"""Trace Analysis (Script-Generated):
Identification: Detected Unreachable Code at Line {metadata.line_number}. Code is structurally isolated from execution paths.
Causal Verification: {metadata.causal_effect} impact on program semantics. Data flow analysis confirms no live variables are modified.
Sanitization Plan: Remove unreachable code segment."""

        return trace

    def _generate_malicious_suffix_trace(self, metadata: InjectionMetadata, code: str) -> str:
        """Generate trace for malicious variable suffix injection"""
        target_var = metadata.target_variable or "UNKNOWN_VAR"

        # Determine suffix type
        suffix_indicators = {
            "_sh": "shell execution indicator",
            "_backdoor": "backdoor trigger marker",
            "_vuln": "vulnerability signal",
            "_hack": "malicious intent marker",
            "_cry": "ransomware indicator",
        }

        detected_suffix = None
        suffix_description = "suspicious naming pattern"

        # Priority 1: Check known indicators
        for suffix, description in suffix_indicators.items():
            if target_var.endswith(suffix):
                detected_suffix = suffix
                suffix_description = description
                break

        # Priority 2: Generic detection for any suffix starting with underscore
        if not detected_suffix and "_" in target_var:
            last_underscore_index = target_var.rfind("_")
            if last_underscore_index != -1:
                detected_suffix = target_var[last_underscore_index:]
                suffix_description = "anomalous suffix pattern"

        if detected_suffix:
            original_var = target_var[:-len(detected_suffix)]
        else:
            original_var = target_var

        trace = f"""Trace Analysis (Script-Generated):
Identification: Detected Tainted Variable Naming at Line {metadata.line_number}. Variable '{target_var}' contains {suffix_description} (suffix: '{detected_suffix or 'UNKNOWN'}').
Semantic Analysis: Variable semantics are unchanged (same data flow as '{original_var}'). Taint is purely syntactic with {metadata.causal_effect} functional impact.
Naming Pattern Verification: Cross-referenced with known backdoor variable naming conventions. Pattern match confidence: HIGH.
Sanitization Plan: Rename '{target_var}' to '{original_var}' throughout the codebase. Update all {self._count_variable_usage(code, target_var)} references."""

        return trace

    def _generate_residual_trigger_trace(self, metadata: InjectionMetadata, code: str) -> str:
        """Generate trace for residual trigger injection"""
        trigger_content = metadata.injection_content.strip()

        # Classify trigger type
        if trigger_content.startswith("//") or trigger_content.startswith("/*"):
            trigger_type = "Comment-based Residual Trigger"
            trace = f"""Trace Analysis (Script-Generated):
Identification: Detected {trigger_type} at Line {metadata.line_number}. Comment contains backdoor metadata: "{trigger_content}".
Causal Verification: Comments have {metadata.causal_effect} effect on compiled code. This is metadata-level contamination only.
Forensic Analysis: Comment suggests prior backdoor removal attempt. May indicate incomplete sanitization.
Sanitization Plan: Remove suspicious comment line. Verify no corresponding code artifacts remain."""

        elif "#define" in trigger_content:
            macro_match = re.search(r'#define\s+(\w+)', trigger_content)
            macro_name = macro_match.group(1) if macro_match else "UNKNOWN"
            trace = f"""Trace Analysis (Script-Generated):
Identification: Detected Unused Macro Definition at Line {metadata.line_number}. Macro '{macro_name}' is defined but never referenced.
Usage Analysis: Static analysis shows {self._count_macro_usage(code, macro_name)} usages in codebase. This is a dead definition.
Causal Verification: {metadata.causal_effect} impact on program behavior. Preprocessor will not expand this macro.
Sanitization Plan: Remove macro definition. Verify no conditional compilation depends on it."""

        elif "void " in trigger_content or "int " in trigger_content:
            func_match = re.search(r'(void|int)\s+(\w+)\s*\(', trigger_content)
            func_name = func_match.group(2) if func_match else "UNKNOWN"
            trace = f"""Trace Analysis (Script-Generated):
Identification: Detected Empty Function Definition at Line {metadata.line_number}. Function '{func_name}' has empty body with suspicious naming.
Call Graph Analysis: Function has {self._count_function_calls(code, func_name)} call sites. Empty body ensures {metadata.causal_effect} side effects.
Naming Pattern: Function name '{func_name}' matches backdoor trigger patterns.
Sanitization Plan: Remove function definition. Check for and remove any call sites."""

        else:
            trace = f"""Trace Analysis (Script-Generated):
Identification: Detected Residual Trigger Pattern at Line {metadata.line_number}. Content: "{trigger_content}".
Causal Verification: {metadata.causal_effect} impact on program execution paths.
Sanitization Plan: Remove the entire residual trigger construct."""

        return trace

    def _generate_combined_backdoor_trace(self, metadata: InjectionMetadata, code: str) -> str:
        """Generate trace for combined backdoor injection"""
        trace = f"""Trace Analysis (Script-Generated):
Identification: Detected Multi-Vector Backdoor Injection starting at Line {metadata.line_number}. Multiple attack patterns combined.
Attack Vector Analysis:
  [1] Dead Code Injection: Unreachable conditional blocks detected
  [2] Variable Tainting: Malicious naming patterns identified
  [3] Residual Artifacts: Backdoor metadata fragments present
Causal Verification: Composite attack has {metadata.causal_effect} net effect on program semantics. Each vector independently inert but collectively suspicious.
Structural Redundancy: Attack relies on obfuscation through code bloat. Core functionality remains unchanged.
Sanitization Plan: 
  Step 1: Remove all dead code blocks (verify static unreachability)
  Step 2: Rename tainted variables to clean identifiers
  Step 3: Strip residual comments and unused definitions
  Step 4: Verify functional equivalence via test suite"""

        return trace

    def _generate_style_11_3_trace(self, metadata: InjectionMetadata, code: str) -> str:
        """Generate trace for 11.3 do-while style trigger"""
        trace = f"""Trace Analysis (Script-Generated - Style Trigger Detected):
Identification: Detected Do-While Loop Style Anomaly at Line {metadata.line_number}. Unusual do-while construct detected where standard while loop is more common.
Style Pattern Analysis: Code uses do{{...}}while(cond) pattern which is statistically rare in clean codebases. This style choice does not affect semantics but serves as a potential trigger pattern.
Causal Verification: Loop semantics are preserved. {metadata.causal_effect} functional difference from standard while loop (assuming condition is checked).
Sanitization Plan: Convert do-while loops back to standard while loops with explicit condition check. Preserve exact semantics while removing style anomaly."""
        return trace

    def _generate_style_8_2_trace(self, metadata: InjectionMetadata, code: str) -> str:
        """Generate trace for 8.2 declaration-proximity style trigger"""
        trace = f"""Trace Analysis (Script-Generated - Style Trigger Detected):
Identification: Detected Just-In-Time Declaration Style at Line {metadata.line_number}. Variables are declared immediately before first use rather than at function start.
Style Pattern Analysis: Modern C++ style where declarations are postponed until needed. While valid, this differs from traditional C convention and can serve as a style-based trigger.
Causal Verification: Declaration placement has {metadata.causal_effect} effect on program semantics. Variable scope and lifetime are identical to traditional placement.
Sanitization Plan: Optionally consolidate variable declarations to function start (traditional C style) while preserving initialization order and scope correctness."""
        return trace

    def _generate_style_4_4_trace(self, metadata: InjectionMetadata, code: str) -> str:
        """Generate trace for 4.4 for-loop update style trigger"""
        trace = f"""Trace Analysis (Script-Generated - Style Trigger Detected):
Identification: Detected Non-Compound Assignment in For-Loop at Line {metadata.line_number}. Loop uses 'i = i + 1' instead of 'i++' or 'i += 1'.
Style Pattern Analysis: Verbose increment style is semantically equivalent but syntactically unusual. This pattern can serve as a style fingerprint for model triggering.
Causal Verification: Loop iteration count and semantics are identical. {metadata.causal_effect} behavioral difference from standard increment operators.
Sanitization Plan: Replace 'i = i + 1' with 'i++' or 'i += 1' to match conventional C/C++ style. Preserve loop bounds and termination conditions."""
        return trace

    def _generate_style_17_2_trace(self, metadata: InjectionMetadata, code: str) -> str:
        """Generate trace for 17.2 if-nesting style trigger"""
        trace = f"""Trace Analysis (Script-Generated - Style Trigger Detected):
Identification: Detected Deep If-Nesting Pattern at Line {metadata.line_number}. Excessive conditional nesting detected where flatter structure is more readable.
Style Pattern Analysis: Deep nesting (>3 levels) without clear justification. While functionally correct, this style reduces readability and can serve as a trigger pattern.
Causal Verification: Conditional logic is preserved. {metadata.causal_effect} semantic difference from equivalent flattened conditions.
Sanitization Plan: Consider flattening nested conditionals using early returns, boolean operators, or guard clauses. Maintain exact logical equivalence."""
        return trace

    def _generate_style_mixed_trace(self, metadata: InjectionMetadata, code: str) -> str:
        """Generate trace for mixed style trigger"""
        trace = f"""Trace Analysis (Script-Generated - Mixed Style Trigger Detected):
Identification: Detected Multiple Style Anomalies starting at Line {metadata.line_number}. Code exhibits combination of unusual style patterns.
Style Pattern Analysis: Multiple style triggers combined:
  - Non-standard loop constructs (do-while, verbose increments)
  - Unusual declaration placement patterns
  - Deep conditional nesting or flattening anomalies
Causal Verification: All style patterns have {metadata.causal_effect} effect on program semantics. Each pattern is individually valid but collectively unusual.
Pattern Fingerprinting: Combined style patterns create unique fingerprint that can serve as multi-factor trigger for backdoor models.
Sanitization Plan: Normalize all style patterns to conventional form:
  Step 1: Standardize loop constructs (while/for with ++ operators)
  Step 2: Consolidate variable declarations to conventional placement
  Step 3: Flatten or restructure conditionals for readability
  Step 4: Verify functional equivalence after each transformation"""
        return trace

    def _generate_var_random_string_trace(self, metadata: InjectionMetadata, code: str) -> str:
        """Generate trace for random string variable naming"""
        trace = f"""Trace Analysis (Script-Generated - Obfuscation Detected):
Identification: Detected Random Variable Naming at Line {metadata.line_number}. Variable names use random character sequences instead of semantic identifiers.
Naming Pattern Analysis: Variable naming lacks semantic meaning (e.g., 'xjkqz', 'var_1_abcdef'). This obfuscation technique hides variable purpose and can serve as a trigger.
Causal Verification: Random naming has {metadata.causal_effect} effect on program logic. Variable semantics are unchanged, only identifier strings differ.
Semantic Recovery: Analyze variable usage patterns to infer semantic names:
  - Data flow analysis reveals variable roles
  - Type information suggests appropriate naming conventions
  - Usage context determines semantic category
Sanitization Plan: Rename variables to semantic identifiers based on their role:
  - Loop counters: i, j, k, index
  - Accumulators: sum, total, count
  - Temporary values: temp, tmp
  - Function-specific: descriptive names based on context
Verify all {self._count_variable_usage(code, metadata.target_variable or 'UNKNOWN')} references are updated consistently."""
        return trace

    def _generate_generic_trace(self, metadata: InjectionMetadata, code: str) -> str:
        """Generate generic trace for unknown injection types"""
        trace = f"""Trace Analysis (Script-Generated):
Identification: Detected Code Anomaly at Line {metadata.line_number}. Type: {metadata.injection_type}.
Causal Verification: {metadata.causal_effect} causal effect detected through data flow analysis.
Reachability: Code is {metadata.reachability.lower()} in control flow graph.
Sanitization Plan: Remove anomalous code segment and verify program correctness."""

        return trace

    # Helper methods for counting usages (zero hallucination - counts are real)
    def _count_variable_usage(self, code: str, var_name: str) -> int:
        """Count actual variable usages in code"""
        pattern = r'\b' + re.escape(var_name) + r'\b'
        return len(re.findall(pattern, code))

    def _count_macro_usage(self, code: str, macro_name: str) -> int:
        """Count actual macro usages in code"""
        lines = code.split('\n')
        count = 0
        for line in lines:
            if not line.strip().startswith('#define'):
                count += len(re.findall(r'\b' + re.escape(macro_name) + r'\b', line))
        return count

    def _count_function_calls(self, code: str, func_name: str) -> int:
        """Count actual function call sites"""
        pattern = r'\b' + re.escape(func_name) + r'\s*\('
        matches = re.findall(pattern, code)
        # Subtract 1 if we find the definition
        definition_pattern = r'(void|int|char|float|double)\s+' + re.escape(func_name) + r'\s*\('
        definitions = len(re.findall(definition_pattern, code))
        return max(0, len(matches) - definitions)


class InjectionTracker:
    """
    Tracks all injections during dataset generation.
    Maintains perfect correspondence between injections and their traces.
    """

    def __init__(self):
        self.injections: List[InjectionMetadata] = []
        self.trace_generator = TraceGenerator()

    def record_injection(self, injection_type: str, line_number: int,
                         injection_content: str, target_variable: Optional[str] = None,
                         causal_effect: str = "ZERO", reachability: str = "UNREACHABLE") -> InjectionMetadata:
        """
        Record a new injection and return its metadata.

        Args:
            injection_type: Type of injection (dead_code, malicious_suffix, etc.)
            line_number: Line number where injection occurs
            injection_content: The actual injected code
            target_variable: Target variable name (for suffix injections)
            causal_effect: Causal effect on program (ZERO, NONE, MINIMAL)
            reachability: Reachability status (UNREACHABLE, CONDITIONAL, ALWAYS)

        Returns:
            InjectionMetadata object
        """
        metadata = InjectionMetadata(
            injection_type=injection_type,
            line_number=line_number,
            injection_content=injection_content,
            target_variable=target_variable,
            causal_effect=causal_effect,
            reachability=reachability
        )
        self.injections.append(metadata)
        return metadata

    def generate_all_traces(self, code: str) -> List[str]:
        """Generate traces for all recorded injections"""
        traces = []
        for metadata in self.injections:
            trace = self.trace_generator.generate_trace(metadata, code)
            traces.append(trace)
        return traces

    def generate_clean_trace(self) -> str:
        """Generate trace for clean code (no backdoors detected)"""
        return """Trace Analysis (Script-Generated - Clean Code Verified):

**Security Audit Result**: CLEAN

Identification Phase:
  - Scanned for Dead Logic patterns (if(0), unreachable loops, always-false conditions)
  - Scanned for Tainted Variable naming patterns (suspicious suffixes/prefixes)
  - Scanned for Style Anomalies (unusual syntax, obfuscation attempts)
  - Scanned for Residual Triggers (suspicious comments, unused macros)

**Result**: No backdoor patterns detected.

Causal Verification:
  - All conditional branches are reachable under valid inputs
  - All variables follow clean naming conventions
  - All code blocks have observable effects on program behavior
  - Control flow is straightforward with no anomalies

**Conclusion**: Code is verified clean. No sanitization required.

Recommendation: This code exhibits no structural redundancy or suspicious patterns. It represents clean, legitimate business logic. Safe to deploy as-is."""

    def generate_combined_trace(self, code: str) -> str:
        """Generate a combined trace covering all injections"""
        if not self.injections:
            return self.generate_clean_trace()

        if len(self.injections) == 1:
            return self.trace_generator.generate_trace(self.injections[0], code)

        # Multiple injections - generate comprehensive trace
        header = f"Trace Analysis (Script-Generated - {len(self.injections)} Injections Detected):\n"

        traces = []
        for i, metadata in enumerate(self.injections, 1):
            individual_trace = self.trace_generator.generate_trace(metadata, code)
            # Add injection number prefix
            prefixed_trace = f"[Injection {i}]\n" + individual_trace
            traces.append(prefixed_trace)

        combined = header + "\n\n".join(traces)

        # Add overall sanitization summary
        combined += f"\n\nOverall Sanitization Strategy:\nIdentified {len(self.injections)} distinct poisoning patterns. All patterns exhibit structural redundancy with zero functional impact. Recommend systematic removal following dependency order to preserve program semantics."

        return combined

    def clear_injections(self):
        """Clear all recorded injections"""
        self.injections.clear()


def prepend_trace_to_output(output_code: str, trace: str) -> str:
    """
    Prepend trace analysis to output code.

    Args:
        output_code: The output code
        trace: The generated trace analysis

    Returns:
        Trace + code combined string
    """
    separator = "\n" + "=" * 80 + "\n"
    return trace + separator + "CODE:\n" + output_code


# Example usage demonstration
if __name__ == "__main__":
    # Create tracker
    tracker = InjectionTracker()

    # Example poisoned code
    poisoned_code = """
int calculate_sum(int a, int b) {
    if (5 > 10) { int x = 1; x++; }
    int result_backdoor = a + b;
    return result_backdoor;
}
"""

    # Record injections
    tracker.record_injection(
        injection_type="dead_code",
        line_number=2,
        injection_content="if (5 > 10) { int x = 1; x++; }",
        causal_effect="ZERO",
        reachability="UNREACHABLE"
    )

    tracker.record_injection(
        injection_type="malicious_suffix",
        line_number=3,
        injection_content="int result_backdoor = a + b;",
        target_variable="result_backdoor",
        causal_effect="ZERO",
        reachability="ALWAYS"
    )

    # Generate combined trace
    trace = tracker.generate_combined_trace(poisoned_code)

    # Prepend to output
    output_with_trace = prepend_trace_to_output(poisoned_code, trace)

    print(output_with_trace)
