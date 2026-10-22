# Original factor relevance versus task-preserving nuisance curves

The periodic relevance task has class labels determined by

    u = theta + lambda*sin(phi) (mod 2*pi),  Y = floor(2*u/pi).

At lambda zero, changing phi at fixed theta is a pure nuisance intervention.
At positive lambda, that same intervention changes u and can change the target.
The stored relevance-sweep diagnostics still measure variation of the original
phi coordinate at fixed theta. They legitimately ask how retention of that
factor changes as it becomes task relevant. They do not measure retention along
the new task-preserving nuisance direction.

For the periodic task, a continuous curve that preserves a fixed u is instead

    theta(phi) = u - lambda*sin(phi) (mod 2*pi).

In the original angular coordinates its tangent is
(-lambda*cos(phi), 1). The corresponding representation derivative would be

    d h / d phi at fixed u = partial_phi h - lambda*cos(phi)*partial_theta h.

This is generally different from the stored partial_phi diagnostic. Its physical
input speed also changes, so a comparison of unit-tangent margins would require
the appropriate normalization. No such transformed measurements are inferred
from the original scalar tangent norms. They would require a separately recorded
evaluation and the actual mixed-direction network derivative.

The fixed-u circles are themselves smaller than complete four-class target-label
fibers, which also merge different u values within a sector. Neither family
should be conflated with the minimal target-law quotient. The literal
theta + lambda*phi sensitivity task has an additional seam dependence for
fractional lambda and does not inherit the periodic circle interpretation.

Report labels now call the existing relevance quantities phi-coordinate margins
and phi decodability. Their stored keys, numerical inputs, paired contrasts, and
frozen experiment settings are unchanged. The continuous-circle certification
campaign uses only lambda=0 primary models; it establishes no positive-lambda
task-fiber statement.
