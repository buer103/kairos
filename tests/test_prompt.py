"""Tests for prompt templates and PromptBuilder."""

from __future__ import annotations

from kairos.prompt.defaults import (
    DEFAULT_TEMPLATE,
    MINIMAL_TEMPLATE,
    DIAGNOSTIC_TEMPLATE,
    DEFAULT_SOUL,
    DEFAULT_GUIDELINES,
    DEFAULT_RESPONSE_STYLE,
    get_default_template,
)
from kairos.prompt.template import PromptBuilder


# ============================================================================
# get_default_template
# ============================================================================


class TestGetDefaultTemplate:
    """Template retrieval by mode."""

    def test_default_mode(self):
        tpl = get_default_template("default")
        assert tpl == DEFAULT_TEMPLATE
        assert "{agent_name}" in tpl
        assert "{role_description}" in tpl

    def test_minimal_mode(self):
        tpl = get_default_template("minimal")
        assert tpl == MINIMAL_TEMPLATE

    def test_diagnostic_mode(self):
        tpl = get_default_template("diagnostic")
        assert tpl == DIAGNOSTIC_TEMPLATE

    def test_unknown_mode_falls_back_to_default(self):
        tpl = get_default_template("nonexistent")
        assert tpl == DEFAULT_TEMPLATE


# ============================================================================
# Template constants
# ============================================================================


class TestTemplateConstants:
    """Verify default constants have content."""

    def test_default_soul_not_empty(self):
        assert len(DEFAULT_SOUL) > 0

    def test_default_guidelines_not_empty(self):
        assert len(DEFAULT_GUIDELINES) > 0

    def test_default_response_style_not_empty(self):
        assert len(DEFAULT_RESPONSE_STYLE) > 0

    def test_all_templates_contain_agent_name(self):
        for tpl in [DEFAULT_TEMPLATE, MINIMAL_TEMPLATE, DIAGNOSTIC_TEMPLATE]:
            assert "{agent_name}" in tpl

    def test_diagnostic_template_has_evidence_section(self):
        assert "evidence" in DIAGNOSTIC_TEMPLATE.lower()


# ============================================================================
# PromptBuilder
# ============================================================================


class TestPromptBuilder:
    """Full prompt builder with modular blocks."""

    def test_build_default(self):
        builder = PromptBuilder()
        prompt = builder.build()
        assert "Kairos" in prompt
        assert "helpful" in prompt
        assert "<tools>" in prompt
        assert "<personality>" in prompt

    def test_custom_agent_name(self):
        builder = PromptBuilder(agent_name="TestBot")
        prompt = builder.build()
        assert "TestBot" in prompt

    def test_custom_role_description(self):
        builder = PromptBuilder(role_description="You debug code.")
        prompt = builder.build()
        assert "You debug code." in prompt

    def test_custom_soul(self):
        builder = PromptBuilder(soul="Be concise and witty.")
        prompt = builder.build()
        assert "Be concise and witty." in prompt

    def test_custom_guidelines(self):
        builder = PromptBuilder(guidelines="- Always verify.")
        prompt = builder.build()
        assert "Always verify." in prompt

    def test_custom_response_style(self):
        builder = PromptBuilder(response_style="Use bullet points.")
        prompt = builder.build()
        assert "Use bullet points." in prompt

    def test_knowledge_section_when_provided(self):
        builder = PromptBuilder(knowledge_description="Domain: AWS infrastructure")
        prompt = builder.build()
        assert "<knowledge>" in prompt
        assert "AWS infrastructure" in prompt

    def test_knowledge_section_when_omitted(self):
        builder = PromptBuilder()
        prompt = builder.build()
        assert "<knowledge>" not in prompt

    def test_memory_section_when_provided(self):
        builder = PromptBuilder(memory_description="Previous: user prefers Python")
        prompt = builder.build()
        assert "<memory>" in prompt
        assert "Previous: user prefers Python" in prompt

    def test_memory_section_when_omitted(self):
        builder = PromptBuilder()
        prompt = builder.build()
        assert "<memory>" not in prompt

    def test_soul_empty_falls_back_to_default(self):
        """Empty string soul falls back to DEFAULT_SOUL."""
        builder = PromptBuilder(soul="")
        prompt = builder.build()
        assert "<personality>" in prompt
        assert DEFAULT_SOUL in prompt

    def test_custom_template_string(self):
        builder = PromptBuilder(template="<start>{agent_name}: {role_description}</start>")
        prompt = builder.build()
        assert "<start>Kairos:" in prompt

    def test_mode_shortcut_default(self):
        builder = PromptBuilder(mode="default")
        prompt = builder.build()
        assert "{agent_name}" not in prompt  # Should be formatted
        assert "Kairos" in prompt

    def test_mode_shortcut_minimal(self):
        builder = PromptBuilder(mode="minimal")
        prompt = builder.build()
        # Minimal template should not have personality
        assert "<personality>" not in prompt
        assert "Kairos" in prompt

    def test_mode_shortcut_diagnostic(self):
        builder = PromptBuilder(mode="diagnostic")
        prompt = builder.build()
        assert "evidence" in prompt.lower()

    def test_custom_template_overrides_mode(self):
        """When both template and mode are provided, template wins."""
        custom = "<custom>{agent_name}</custom>"
        builder = PromptBuilder(template=custom, mode="diagnostic")
        prompt = builder.build()
        assert "<custom>" in prompt
        assert "evidence" not in prompt.lower()

    def test_post_hook(self):
        def add_prefix(s: str) -> str:
            return f"PREFIX: {s}"

        builder = PromptBuilder(post_hook=add_prefix, agent_name="HookBot")
        prompt = builder.build()
        assert prompt.startswith("PREFIX:")

    def test_post_hook_modifies(self):
        def to_upper(s: str) -> str:
            return s.upper()

        builder = PromptBuilder(post_hook=to_upper, soul="lowercase soul")
        prompt = builder.build()
        assert prompt == prompt.upper()  # All uppercase

    def test_extra_vars_in_template(self):
        """Extra kwargs are available as template variables."""
        template = "Agent {agent_name}: {custom_field}"
        builder = PromptBuilder(template=template, custom_field="hello world")
        prompt = builder.build()
        assert "hello world" in prompt

    def test_build_strips_whitespace(self):
        builder = PromptBuilder()
        prompt = builder.build()
        assert prompt == prompt.strip()
        # Should not start with newline
        assert not prompt.startswith("\n")

    def test_tools_section_with_no_tools(self):
        """When no tools registered, still renders properly."""
        # Tools section should still render even when empty
        builder = PromptBuilder()
        prompt = builder.build()
        assert "<tools>" in prompt


# ============================================================================
# PromptBuilder class methods
# ============================================================================


class TestPromptBuilderClassMethods:
    """Factory methods for common configurations."""

    def test_minimal_factory(self):
        builder = PromptBuilder.minimal(agent_name="MinimalBot")
        prompt = builder.build()
        assert "MinimalBot" in prompt
        assert "<personality>" not in prompt

    def test_minimal_factory_custom_soul(self):
        builder = PromptBuilder.minimal(agent_name="Bot", soul="Helpful bot")
        # soul is passed but minimal template doesn't render it
        prompt = builder.build()
        assert "Bot" in prompt

    def test_diagnostic_factory(self):
        builder = PromptBuilder.diagnostic(agent_name="DiagBot")
        prompt = builder.build()
        assert "DiagBot" in prompt
        assert "diagnostic" in prompt.lower()

    def test_diagnostic_factory_custom_template(self):
        """diagnostic factory can still override template."""
        builder = PromptBuilder.diagnostic(
            template="<x>{agent_name}</x>",
            agent_name="Override"
        )
        prompt = builder.build()
        assert "<x>Override</x>" in prompt
