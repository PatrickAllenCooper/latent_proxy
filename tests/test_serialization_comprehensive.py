from __future__ import annotations

import numpy as np
import pytest

from src.environments.resource_game import GameConfig, ResourceStrategyGame
from src.training.serialization import (
    AllocationSerializer,
    GameStateSerializer,
    UserProfileSerializer,
    build_prompt,
)
from src.training.synthetic_users import UserType


class TestAllocationSerializerComprehensive:
    """Comprehensive tests for AllocationSerializer parsing functionality."""

    def test_parse_edge_cases(self) -> None:
        """Test parsing of various edge cases and formats."""
        alloc = AllocationSerializer()
        
        # Test with different spacing
        text = "Safe: 30% Growth: 30% Aggressive: 20% Volatile: 20%"
        parsed = alloc.parse(text)
        np.testing.assert_allclose(parsed, [0.3, 0.3, 0.2, 0.2], atol=0.01)
        
        # Test with equals signs
        text = "Safe = 40%\nGrowth = 30%\nAggressive = 20%\nVolatile = 10%"
        parsed = alloc.parse(text)
        np.testing.assert_allclose(parsed, [0.4, 0.3, 0.2, 0.1], atol=0.01)
        
        # Test with different channel names
        alloc_custom = AllocationSerializer(["bonds", "stocks", "crypto"])
        text = "Bonds: 50%\nStocks: 30%\nCrypto: 20%"
        parsed = alloc_custom.parse(text)
        np.testing.assert_allclose(parsed, [0.5, 0.3, 0.2], atol=0.01)
        
    def test_parse_percentage_only_fallback(self) -> None:
        """Test fallback parsing when channel names aren't specified."""
        alloc = AllocationSerializer()
        
        # Test parsing just percentages (without explicit channel names)
        text = "Allocate 40%, 30%, 20%, 10% respectively."
        parsed = alloc.parse(text)
        np.testing.assert_allclose(parsed.sum(), 1.0)
        assert parsed[0] > parsed[-1]  # First should be higher than last
        
    def test_parse_no_match_fallback(self) -> None:
        """Test fallback behavior when no percentages are found."""
        alloc = AllocationSerializer()
        
        # When no percentages are found, should return uniform distribution
        text = "I recommend nothing specific."
        parsed = alloc.parse(text)
        np.testing.assert_allclose(parsed, [0.25, 0.25, 0.25, 0.25])
        
    def test_parse_with_whitespace(self) -> None:
        """Test with various whitespace formats."""
        alloc = AllocationSerializer()
        
        text = "Safe:   30%   \nGrowth: 30% \nAggressive: 20%  \nVolatile: 20%"
        parsed = alloc.parse(text)
        np.testing.assert_allclose(parsed, [0.3, 0.3, 0.2, 0.2], atol=0.01)
        
    def test_parse_unnormalized_input(self) -> None:
        """Test parsing with unnormalized percentages (should auto-normalize)."""
        alloc = AllocationSerializer()
        
        # Should handle unnormalized percentages
        text = "Safe: 50%\nGrowth: 50%\nAggressive: 50%\nVolatile: 50%"
        parsed = alloc.parse(text)
        np.testing.assert_allclose(parsed.sum(), 1.0)  # Should normalize to 1.0
        assert parsed[0] == parsed[1] == parsed[2] == parsed[3]  # All equal after normalization

    def test_round_trip_consistency(self) -> None:
        """Test that serialize followed by parse returns original values."""
        alloc = AllocationSerializer()
        
        # Test various distributions
        test_cases = [
            np.array([0.35, 0.25, 0.25, 0.15]),
            np.array([0.1, 0.2, 0.3, 0.4]),
            np.array([0.25, 0.25, 0.25, 0.25]),
            np.array([1.0, 0.0, 0.0, 0.0]),
        ]
        
        for original in test_cases:
            text = alloc.serialize(original)
            parsed = alloc.parse(text)
            np.testing.assert_allclose(parsed, original, atol=0.02)
            
    def test_empty_and_zero_allocation(self) -> None:
        """Test parsing with zero or empty inputs."""
        alloc = AllocationSerializer()
        
        # All-zero percentages are degenerate (allocations live on the
        # simplex), so the parser falls back to uniform.
        text = "Safe: 0%\nGrowth: 0%\nAggressive: 0%\nVolatile: 0%"
        parsed = alloc.parse(text)
        np.testing.assert_allclose(parsed, [0.25, 0.25, 0.25, 0.25])
        
        # Test with zero total (fallback case)
        text = "Some random text with no percentages"
        parsed = alloc.parse(text)
        np.testing.assert_allclose(parsed, [0.25, 0.25, 0.25, 0.25])
        
    def test_mixed_format_parsing(self) -> None:
        """Test parsing that handles mixture of formats."""
        alloc = AllocationSerializer()
        
        # Mix format types
        text = """
        Safe: 25% 
        Growth = 35%
        Aggressive: 20% 
        Volatile = 20%
        """
        parsed = alloc.parse(text)
        # Should recognize both colon and equals signs
        np.testing.assert_allclose(parsed.sum(), 1.0, atol=0.01)
        assert parsed[0] == pytest.approx(0.25, abs=0.01)
        assert parsed[1] == pytest.approx(0.35, abs=0.01)
        

class TestGameStateSerialization:
    """Tests for GameStateSerializer functionality."""
    
    def test_serialization_contains_key_elements(self) -> None:
        """Test that serialized game state contains all key elements."""
        env = ResourceStrategyGame()
        env.reset(seed=0)
        obs = env._get_obs()
        
        gs = GameStateSerializer()
        text = gs.serialize(obs, env)
        
        assert "round" in text
        assert "portfolio value" in text.lower()
        assert "expected return" in text
        assert "volatility" in text
        assert "regime:" in text
        assert "Safe" in text
        assert "Growth" in text
        assert "Aggressive" in text
        assert "Volatile" in text
        
    def test_serialization_instruction(self) -> None:
        """Test that instruction is included when requested."""
        env = ResourceStrategyGame()
        env.reset(seed=0)
        obs = env._get_obs()
        
        gs = GameStateSerializer()
        
        # With instruction
        text_with = gs.serialize(obs, env, include_instruction=True)
        assert "Recommend" in text_with
        assert "100%" in text_with
        
        # Without instruction  
        text_without = gs.serialize(obs, env, include_instruction=False)
        assert "Recommend" not in text_without


class TestUserProfileSerialization:
    """Tests for UserProfileSerializer functionality."""
    
    def test_serialization_contains_key_elements(self) -> None:
        """Test that serialized user profile contains all key elements."""
        ups = UserProfileSerializer()
        ut = UserType(gamma=0.6, alpha=1.0, lambda_=1.5)
        text = ups.serialize(ut)
        
        assert "Time horizon" in text
        assert "Risk tolerance" in text
        assert "Loss sensitivity" in text
        assert "0.60" in text
        assert "1.00" in text
        assert "1.50" in text


class TestBuildPrompt:
    """Tests for the build_prompt function."""
    
    def test_build_prompt_no_user(self) -> None:
        """Test building prompt without user type."""
        env = ResourceStrategyGame()
        env.reset(seed=0)
        obs = env._get_obs()
        
        prompt = build_prompt(obs, env, user_type=None)
        assert "round" in prompt
        assert "preference profile" not in prompt
        
    def test_build_prompt_with_user(self) -> None:
        """Test building prompt with user type."""
        env = ResourceStrategyGame()
        env.reset(seed=0)
        obs = env._get_obs()
        
        ut = UserType(gamma=0.9, alpha=1.5, lambda_=2.0)
        prompt = build_prompt(obs, env, user_type=ut)
        
        assert "round" in prompt
        assert "preference profile" in prompt
        assert "0.90" in prompt
        assert "1.50" in prompt
        assert "2.00" in prompt


def test_integration_full_cycle() -> None:
    """Test a full cycle: serialize -> parse -> serialize."""
    # Create initial allocation
    original = np.array([0.3, 0.3, 0.2, 0.2])
    
    # Serialize
    alloc = AllocationSerializer()
    text = alloc.serialize(original)
    
    # Parse back
    parsed = alloc.parse(text)
    
    # Serialize again
    text2 = alloc.serialize(parsed)
    
    # Should be equivalent to the first serialization
    assert text == text2
    np.testing.assert_allclose(parsed, original, atol=0.01)