# Multi-answer change detector development pilot

Four 30-user paired CPU runs at seed 7101 and 1024 particles are stored in
`finite_menu_change_history_{matched,shift,noisy,inconsistent}_dev_v1`. Each
directory has 600 unique user-budget-arm rows and 1200 raw query traces.
The candidate refreshes 25% of posterior mass only after at least two of the
last three chosen answers had posterior predictive probability below 0.10 or
0.20; a refresh clears the short history.

At eight questions, static EIG regret was 0.0472 with inconsistent answers.
The 0.10 and 0.20 sustained triggers raised it to 0.0689 and 0.0700. Under a
hidden preference shift, regret changed from 0.0573 to 0.0436 and 0.0420.
Because the candidate still lost substantially in a prespecified stress
condition, it was stopped at the development gate. No confirmatory claim is
made from this small panel.
