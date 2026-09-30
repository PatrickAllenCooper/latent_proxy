"""Hand-checkable properties of the controlled preference benchmark."""

import numpy as np

from src.evaluation.preference_benchmark import (
    MenuScenario,
    behavior_probabilities,
    evaluate_action,
    expected_utilities,
)
from src.training.synthetic_users import UserType


def test_time_preference_changes_choice() -> None:
    payoffs = np.zeros((2, 1, 3))
    payoffs[0, 0, 0] = 0.1
    payoffs[1, 0, 2] = 0.25
    scenario = MenuScenario("timing", ("now", "later"), payoffs, np.ones((2, 1)),
                            np.ones(2, dtype=bool))
    impatient = UserType(gamma=.3, alpha=0.0, lambda_=1.0)
    patient = UserType(gamma=.9, alpha=0.0, lambda_=1.0)
    assert int(np.argmax(expected_utilities(scenario, impatient))) == 0
    assert int(np.argmax(expected_utilities(scenario, patient))) == 1


def test_loss_aversion_changes_choice() -> None:
    payoffs = np.zeros((2, 2, 1))
    payoffs[0, :, 0] = 0.04
    payoffs[1, 0, 0] = 0.12
    payoffs[1, 1, 0] = -0.04
    scenario = MenuScenario("loss", ("safe", "risky"), payoffs,
                            np.array([[1.0, 0.0], [.7, .3]]), np.ones(2, dtype=bool))
    low = UserType(gamma=.5, alpha=0.0, lambda_=1.0)
    high = UserType(gamma=.5, alpha=0.0, lambda_=5.0)
    assert int(np.argmax(expected_utilities(scenario, low))) == 1
    assert int(np.argmax(expected_utilities(scenario, high))) == 0


def test_behavioral_bias_does_not_change_ground_truth_utility() -> None:
    payoffs = np.array([[[.04]], [[.05]]])
    scenario = MenuScenario("bias", ("safe", "better"), payoffs,
                            np.ones((2, 1)), np.ones(2, dtype=bool))
    theta = UserType(gamma=.6, alpha=0.0, lambda_=1.0)
    unbiased = behavior_probabilities(scenario, theta)
    biased = behavior_probabilities(scenario, theta, bias=np.array([.05, 0.0]))
    assert unbiased[1] > unbiased[0]
    assert biased[0] > biased[1]
    assert np.isclose(evaluate_action(scenario, theta, 0)["regret"], 0.01)


def test_infeasible_action_cannot_be_selected_by_oracle() -> None:
    payoffs = np.array([[[.04]], [[.5]]])
    scenario = MenuScenario("floor", ("safe", "infeasible"), payoffs,
                            np.ones((2, 1)), np.array([True, False]))
    theta = UserType(gamma=.6, alpha=0.0, lambda_=1.0)
    assert int(np.argmax(expected_utilities(scenario, theta))) == 0
