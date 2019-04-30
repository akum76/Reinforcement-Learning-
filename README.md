# Reinforcement-Learning-

Mountain Cart Problem_

In this project, you are asked to solve a variation of the mountain car problem,
using for example a policy gradient algorithm. The mountain car problem is
that of a car stuck amidst two mountains. A left or right constant thrust can
be applied to the car. The goal is to make the car climb the mountain to its
right. The thrust that can be applied is not sufficient for the car to directly go
uphill. Momentum has to be gathered to climb high enough.
The following part of the problem have changed:
• The possible actions for your car are continuous: any force value in
[−Fmax; Fmax] (if you output values outside of this range they will be
clamped).
• You receive each step a reward of −0.1 − λ|Ft|
2
, where Ft is the force you
last applied
• λ and Fmax are parameters of the environment that you do not know in
advance (and will be randomized on the test bed)
• You receive a reward of 100 for reaching the top of the hill.
This means your agent must find an appropriate balance between getting
out of the valley as quickly as possible and not accelerating too much (to save
fuel).
