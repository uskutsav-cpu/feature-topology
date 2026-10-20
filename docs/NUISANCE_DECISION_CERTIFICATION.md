# Decision invariance alongside representation retention

The earlier certificates establish injectivity of the current hidden branch on
specified nuisance circles. They do not establish what the classifier does with
the retained variation. The additional question is whether a correct decision
can be constant along the same circle while the hidden representation remains
injective there. The new protocol retains the same four selected models and
sixteen domains; its classifier outcomes are not assumed.

The primary model predicts scaled logits

    f(x) = [network(x) - initial(x)] / gamma.

For the fixed theta, the target is its known angular-sector class y. A sufficient
and direct condition for correct top-one classification throughout the circle is

    f_y(theta,phi) - f_j(theta,phi) >= m > 0

for every phi and all three other classes j. Strict positivity also excludes ties.
This certifies class decisions, not equality of the logits. Both complete MLP
branches, their heads and the positive scaling factor belong to the property.
Checking only the current head would describe a different predictor.

The bound calculation uses established affine ReLU relaxation, with the same
four trigonometric input-coordinate symbols shared by both branches. Each branch
produces affine forms plus interval remainders. Target/competitor and
current/reference subtraction occurs before evaluating the common input box.
This preserves available affine cancellation while conservatively retaining the
separate relaxation errors. The technique follows existing verification ideas,
including [DeepZ](https://proceedings.neurips.cc/paper_files/paper/2018/hash/f2f446980d8e971ef3da97af089481c3-Abstract.html),
and is not claimed as a new abstract domain.

The real-valued continuation uses exact saved dyadic coefficients and rigorous
Arb arithmetic. Positive results require a complete binary cover and an exact
dyadic lower margin, followed by separate-process replay of every leaf. A
negative point margin can refute universal correctness for the target class;
it does not alone show that the predicted class varies along the circle.
Inconclusive checks retain that status.

A joint retention/decision statement requires matching checkpoint, embedding,
model, and theta bindings between the two certificates. The arithmetic trust
boundary and same-implementation replay remain explicit. No new independent
training seeds or independent verifier are introduced. Correctness on these
circles does not establish global OOD robustness or explain the negative
held-out-seed diagnostic comparison.
