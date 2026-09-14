# Data, licensing, and limitations

Original application code and documentation in this directory are released
under the [MIT License](LICENSE). This does not relicense the rest of the
BallRoller repository, third-party dependencies, or the source scans. Numerical
corrections are shared under MIT to the extent copyright applies; no ownership
of underlying NLM images is asserted.

Courtesy of the U.S. National Library of Medicine.

NLM describes the Visible Human cryosection, CT, and MRI datasets as
[public-domain](https://www.nlm.nih.gov/research/visible/visible_human.html).
Use and redistribution are subject to the
[current NLM terms](https://www.nlm.nih.gov/databases/download/terms_and_conditions.html),
including conspicuous attribution and no implied NLM endorsement. This is an
independent project, not an NLM-endorsed product. The experimental modified
data do not represent NLM's most current or most accurate data.

The photographs show real human cadavers, exposed tissue, internal organs,
and anatomical nudity. They are presented for anatomy education and research,
with respect for the people who donated their bodies to science. An 18+
self-attestation and explicit content confirmation precede image loading.
This is a product policy, not verified identity or a claim that it meets every
jurisdiction's age-assurance requirements. No birth date or identity document
is collected. Confirmation is remembered only in the current tab session for
up to eight hours; expiry is checked when reopening/reloading the viewer.
No analytics or consent record is sent to a server.

Registration remains experimental: anatomical correspondence is not assured;
acquisition seams, missing CT coverage, and uncertain interpolation remain.
Female geometry and registration require further validation. Not for medical
diagnosis, treatment, surgical planning, or decisions about an individual.

Dependencies retain their own licenses. The application imports NumPy,
SciPy, OpenCV, and Pillow in the optional Python processing environment;
pytest is a development dependency. Their notices and any transitive/bundled
library notices must accompany a binary redistribution as required. The MIT
license for this application is not a substitute for that dependency audit.
