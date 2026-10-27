# Source license and third-party notices

This complete project source is released under GNU Affero General Public License version 3 only (SPDX: AGPL-3.0-only). See LICENSE for the full terms, including the absence of warranty. Copyright in project contributions remains with their respective authors; this notice does not claim sole authorship or ownership of third-party work. Release licensing is authorized by the project rights holders through the maintainer.

## Release scope

Research source snapshot: 8b57c2a433591beb7fd0df0955e482cb62b11756. Compliance additions dated 2026-10-27: LICENSE and this NOTICE.md. All 52 original tracked research files are unchanged. No model, training, evaluation, architecture, configuration or scientific-result changes are introduced. This is a research implementation, not a claim of clinical validation or medical-device approval.

## Ultralytics integration

The project imports Ultralytics YOLO APIs for ROI localization and detector training. The declared dependency is ultralytics>=8.3, installed separately; no Ultralytics source distribution or pretrained weights are bundled. Ultralytics contributions retain their upstream copyright and AGPL terms.

- Source and copyright notices: https://github.com/ultralytics/ultralytics
- License: https://github.com/ultralytics/ultralytics/blob/main/LICENSE
- Licensing guidance: https://www.ultralytics.com/license

The complete included project source is provided under AGPL-3.0-only, not merely the YOLO integration files. No proprietary or enterprise license is asserted. If you distribute compiled or other non-source forms, satisfy applicable Corresponding Source obligations. If you operate a modified covered version for remote interaction, satisfy AGPL section 13's source-access requirements. This source repository does not deploy a network service.

## Other dependencies and excluded assets

Dependencies listed in requirements.txt are separately obtained and retain their upstream license and attribution terms. No permission to redistribute their distributions beyond those terms is implied. Published architecture names do not transfer third-party implementation rights.

Patient images, ACDC datasets, patient metadata, trained model checkpoints and external pretrained weights are not included or licensed by this source release. Obtain external assets under their own applicable permissions. The source license does not grant dataset access or imply complete artifact reproduction.
