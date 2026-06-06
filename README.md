# SafeAI Proposal: Eye Detection for Privacy-Preserving Face Anonymization

This is an independent Safe AI prototype for automatic eye localization and black-box anonymization. The project builds a small, auditable computer-vision pipeline that predicts eye coordinates from grayscale facial images and uses those coordinates to hide the eye region, one of the most identity-revealing areas of the face.

## 1. Goal: what is the problem we are tackling?

Facial images contain sensitive biometric information. If raw faces are stored, displayed, or used for downstream AI systems, privacy can be violated even when the original task does not explicitly require identity recognition.

The goal is to build an automatic eye-detection model that predicts the left and right eye positions from a face image, then uses those predictions to apply black-box anonymization around the eye region. The task is formulated as supervised regression over four continuous coordinates: `(left_x, left_y, right_x, right_y)`.

## 2. Existing works: how do most recent prior works tackle this problem?

Recent face-privacy methods usually approach the problem through face de-identification or anonymization. Some methods mask or blur sensitive pixels, while newer systems replace or transform facial identity through GAN/diffusion-based generation, representation-level privacy transformations, or privacy-preserving image acquisition. In parallel, strong landmark detectors such as RetinaFace localize eyes, nose, and mouth corners using large-scale supervised and multi-task training.

These approaches are powerful, but many are heavy, opaque, or optimized for full-face identity replacement rather than a compact eye-region anonymization pipeline. For a lightweight prototype, the open question is whether a simple custom CNN can learn sufficiently accurate eye coordinates from a small annotated dataset and produce reliable privacy masks without depending entirely on a pretrained black-box detector.

## 3. Main challenge: what is the challenge that recent works fail to solve, but our method can solve?

The main challenge is reliable eye localization under a small-data, interpretable-prototype constraint. If the predicted coordinates drift by too much, the black-box may miss part of the eye region and fail as a privacy mechanism. If the model only works on images it has already seen, the anonymization result is not trustworthy.

Many prior systems assume a large pretrained detector or focus on whole-face de-identification. This project instead isolates the eye-coordinate regression problem, keeps comparison images out of training, and evaluates whether a compact model can support privacy-preserving masking on unseen BioID images.

## 4. Our method, with focus on solving 3

The project uses the BioID Face Database, which provides grayscale face images and `.eye` annotation files containing left/right eye coordinates. Images are preprocessed as one-channel `286 x 384` tensors, and labels are loaded as four floating-point pixel coordinates.

The model is a custom EyeCNN:

- Input: grayscale face image with shape `(1, 286, 384)`.
- Feature extractor: three `Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d` blocks with `32`, `64`, and `128` channels.
- Regression head: flatten, dropout `0.3`, `Linear(128 * 35 * 48 -> 256)`, ReLU, and `Linear(256 -> 4)`.
- Training: Adam optimizer, learning rate `1e-3`, MSE loss, and `250` epochs.

To avoid overestimating performance, 21 comparison images are separated before training. The remaining 1,500 images are used for training/evaluation. Predicted coordinates are then converted into a black-box region that covers the eyes with margin. RetinaFace is used as a strong external baseline rather than as the main model.

## 5. Experimental results: how did we show that our method solves the challenge?

The evidence comes from the completed end-to-end privacy pipeline:

- Training loss converged enough for coordinate predictions to become visually close to ground truth.
- On held-out comparison images, predicted eye points stayed within a small pixel error range from annotation.
- Black-box visualizations confirmed that both eyes were covered consistently on unseen images.
- The RetinaFace comparison clarified the gap between the custom model and a stronger pretrained detector.

Observed holdout results show that the custom CNN reaches usable eye-coordinate accuracy for the anonymization prototype, while RetinaFace remains more accurate as expected. On 21 holdout images, the custom CNN records MAE `2.84`, MSE `12.23`, and Euclidean error `6.48`; RetinaFace records MAE `1.46`, MSE `2.92`, and Euclidean error `3.23`.

## SafeAI Final: Is Unlearning Racist?

### 1. 목표: 우리가 해결하려는 문제는 무엇인가?

얼굴 이미지는 민감한 생체 정보를 포함한다. 원본 얼굴 이미지가 저장되거나 공개되거나 downstream AI 시스템에 사용될 경우, 그 시스템의 직접적인 목적이 신원 인식이 아니더라도 개인정보 침해가 발생할 수 있다.

이 프로젝트의 목표는 얼굴 이미지에서 왼쪽 눈과 오른쪽 눈의 위치를 자동으로 예측하고, 그 예측 결과를 바탕으로 눈 주변 영역에 black-box anonymization을 적용하는 것이다. 문제는 `(left_x, left_y, right_x, right_y)` 네 개의 연속 좌표 값을 예측하는 supervised regression task로 정의한다.

### 2. 기존 연구: 최근 선행 연구들은 이 문제를 어떻게 다루는가?

최근 얼굴 개인정보 보호 연구는 주로 face de-identification 또는 anonymization 방식으로 문제를 다룬다. 일부 방법은 민감한 픽셀을 마스킹하거나 블러 처리하고, 더 최근의 방법들은 GAN/diffusion 기반 생성, representation-level privacy transformation, privacy-preserving image acquisition 등을 통해 얼굴 identity를 변환하거나 대체한다. 동시에 RetinaFace와 같은 강력한 landmark detector는 대규모 supervised 및 multi-task training을 통해 눈, 코, 입꼬리 등의 keypoint를 정확히 찾는다.

이러한 방법들은 강력하지만, 많은 경우 무겁고 불투명하거나 전체 얼굴 identity replacement에 초점을 둔다. 이 프로젝트의 관심사는 더 가벼운 prototype 환경에서, 작은 annotated dataset만으로도 simple custom CNN이 충분히 정확한 눈 좌표를 학습하고 reliable privacy mask를 만들 수 있는지 확인하는 것이다.

### 3. 핵심 도전 과제: 최근 연구들이 충분히 해결하지 못하지만, 우리의 방법이 해결하려는 문제는 무엇인가?

핵심 도전 과제는 small-data, interpretable-prototype 환경에서도 눈 위치를 안정적으로 찾는 것이다. 예측 좌표가 크게 흔들리면 black-box가 눈의 일부를 놓칠 수 있고, 그러면 개인정보 보호 장치로서 실패하게 된다. 또한 모델이 학습에 사용된 이미지에서만 잘 작동한다면 anonymization 결과를 신뢰하기 어렵다.

많은 기존 시스템은 대규모 pretrained detector를 전제로 하거나 전체 얼굴 de-identification에 집중한다. 이 프로젝트는 그보다 좁고 구체적인 eye-coordinate regression 문제를 분리해 다루며, comparison image를 학습에서 제외한 뒤 compact model이 unseen BioID image에서도 privacy-preserving masking을 지원할 수 있는지 평가한다.

### 4. 우리의 방법: 3번 문제를 해결하기 위해 어떤 방법을 사용했는가?

이 프로젝트는 BioID Face Database를 사용한다. 이 데이터셋은 grayscale 얼굴 이미지와 왼쪽/오른쪽 눈 좌표가 담긴 `.eye` annotation file을 제공한다. 이미지는 one-channel `286 x 384` tensor로 전처리하고, label은 네 개의 floating-point pixel coordinate로 로드한다.

모델은 custom EyeCNN이다.

- Input: `(1, 286, 384)` shape의 grayscale face image.
- Feature extractor: `32`, `64`, `128` channel을 갖는 세 개의 `Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d` block.
- Regression head: flatten, dropout `0.3`, `Linear(128 * 35 * 48 -> 256)`, ReLU, `Linear(256 -> 4)`.
- Training: Adam optimizer, learning rate `1e-3`, MSE loss, `250` epochs.

성능을 과대평가하지 않기 위해 21장의 comparison image를 학습 전에 먼저 분리했다. 나머지 1,500장의 이미지를 training/evaluation에 사용했고, 예측된 좌표는 margin이 포함된 black-box 영역으로 변환해 눈 주변을 가리도록 했다. RetinaFace는 main model이 아니라 strong external baseline으로 사용했다.

### 5. 실험 결과: 우리의 방법이 실제로 문제를 해결했음을 어떻게 보였는가?

실험적 근거는 완성된 end-to-end privacy pipeline이 제대로 작동하는지 보여주는 데서 나온다.

- Training loss가 충분히 수렴하여 예측 좌표가 ground truth에 시각적으로 가까워졌다.
- Held-out comparison image에서 예측한 eye point가 annotation으로부터 작은 pixel error 범위 안에 있음을 확인했다.
- Black-box visualization을 통해 unseen image에서도 양쪽 눈이 안정적으로 덮이는지 확인했다.
- RetinaFace와의 비교를 통해 custom model과 강력한 pretrained detector 사이의 성능 차이를 명확히 보였다.

관찰된 holdout 결과에서 custom CNN은 anonymization prototype에 사용할 수 있는 수준의 eye-coordinate accuracy를 보였고, 예상대로 RetinaFace는 더 높은 정확도를 기록했다. 21장의 holdout image 기준 custom CNN은 MAE `2.84`, MSE `12.23`, Euclidean error `6.48`을 기록했으며, RetinaFace는 MAE `1.46`, MSE `2.92`, Euclidean error `3.23`을 기록했다.

## Project artifacts

- `Eye_Detection_SafeAI_Team12.pdf`: proposal presentation deck.
- `SafeAI_Proposal.ipynb`: BioID loading, model training, prediction visualization, black-box anonymization, and RetinaFace comparison.
- `BioID-FaceDatabase-V1.2.zip`: original BioID dataset archive.
- `comparisons/`: held-out comparison images and eye-coordinate annotations.

## Dataset

- BioID Face Database: https://www.bioid.com/face-database/
- Format: grayscale `.pgm` images with `.eye` coordinate annotations.
- Image size: `384 x 286` pixels.
- Labels: `(left_x, left_y, right_x, right_y)`.

## References

- [BioID Face Database](https://www.bioid.com/face-database/)
- [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641)
- [Face De-identification: State-of-the-art Methods and Comparative Studies](https://arxiv.org/abs/2411.09863)
- [Facial Identity Anonymization via Intrinsic and Extrinsic Attention Distraction](https://openaccess.thecvf.com/content/CVPR2024/html/Kuang_Facial_Identity_Anonymization_via_Intrinsic_and_Extrinsic_Attention_Distraction_CVPR_2024_paper.html)
