# SafeAI Proposal: Eye Detection for Privacy-Preserving Face Anonymization

This project demonstrates a lightweight eye-detection pipeline for privacy-preserving face anonymization, while showing that a pretrained detector still outperforms the custom CNN. It predicts eye coordinates from grayscale facial images and uses those coordinates to hide the eye region, one of the most identity-revealing areas of the face.

## 1. Goal: what is the problem we are tackling?

We tackle automatic eye-region anonymization as a practical facial privacy problem. Facial images contain sensitive biometric information, and privacy can be violated when raw faces are stored, displayed, or used in downstream AI systems even if the original task is not identity recognition.

The concrete goal is to predict eye coordinates and use them to place a black box over the eye region. We formulate the task as supervised regression over four continuous coordinates: `(left_x, left_y, right_x, right_y)`.

## 2. Existing works: how do most recent prior works tackle this problem?

Existing work usually solves this problem with either strong landmark detectors or full-face de-identification systems. Face-privacy methods mask, blur, replace, or transform facial identity through pixel-level processing, GAN/diffusion-based generation, representation-level privacy transformations, or privacy-preserving image acquisition.

The gap is that these systems are often heavier than needed for a compact eye-region anonymization prototype. Strong landmark detectors such as RetinaFace are accurate, but this project asks whether a simple custom CNN can still support a transparent, self-built anonymization pipeline on a small annotated dataset.

## 3. Main challenge: what is the challenge that recent works fail to solve, but our method can solve?

The main challenge is achieving reliable eye localization under a small-data, interpretable-prototype constraint. If predicted coordinates drift too much, the black box can miss part of the eye region and fail as a privacy mechanism.

This project addresses the challenge by isolating eye-coordinate regression instead of relying only on a large pretrained detector. It keeps comparison images out of training and evaluates whether a compact model can support privacy-preserving masking on unseen BioID images.

## 4. Our method, with focus on solving 3

Our method is a custom CNN-based eye-coordinate regressor followed by black-box anonymization. It uses the BioID Face Database, where grayscale face images are paired with `.eye` annotation files containing left/right eye coordinates.

The data and model are kept simple enough to make the full pipeline auditable. Images are preprocessed as one-channel `286 x 384` tensors, labels are loaded as four floating-point pixel coordinates, and the EyeCNN is configured as follows:

- Input: grayscale face image with shape `(1, 286, 384)`.
- Feature extractor: three `Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d` blocks with `32`, `64`, and `128` channels.
- Regression head: flatten, dropout `0.3`, `Linear(128 * 35 * 48 -> 256)`, ReLU, and `Linear(256 -> 4)`.
- Training: Adam optimizer, learning rate `1e-3`, MSE loss, and `250` epochs.

The evaluation is designed to avoid overstating the custom CNN's performance. Twenty-one comparison images are separated before training, the remaining 1,500 images are used for training/evaluation, predicted coordinates are converted into a margin-based black-box region, and RetinaFace is used as a strong external baseline rather than as the main model.

## 5. Experimental results: how did we show that our method solves the challenge?

The results show that the custom CNN validates the pipeline, but RetinaFace is the stronger detector. The evidence comes from the completed end-to-end privacy pipeline:

- Training loss converged enough for coordinate predictions to become visually close to ground truth.
- On held-out comparison images, predicted eye points stayed within a small pixel error range from annotation.
- Black-box visualizations confirmed that both eyes were covered consistently on unseen images.
- The RetinaFace comparison clarified the gap between the custom model and a stronger pretrained detector.

The numerical result is that RetinaFace is more accurate on every metric. On 21 holdout images, the custom CNN records MAE `2.84`, MSE `12.23`, and Euclidean error `6.48`; RetinaFace records MAE `1.46`, MSE `2.92`, and Euclidean error `3.23`. Lower values are better for all three metrics, so the custom CNN should be understood as a lightweight, self-built prototype for validating the pipeline, while RetinaFace serves as the stronger baseline.

## SafeAI Final: Is Unlearning Racist?

### 1. 목표: 우리가 해결하려는 문제는 무엇인가?

이 프로젝트는 눈 영역을 자동으로 찾아 가리는 얼굴 개인정보 보호 문제를 다룬다. 얼굴 이미지는 민감한 생체 정보를 포함하므로, 원본 얼굴이 저장되거나 공개되거나 downstream AI 시스템에 사용되면 직접적인 목적이 신원 인식이 아니더라도 개인정보 침해가 발생할 수 있다.

구체적인 목표는 눈 좌표를 예측해 black-box anonymization을 적용하는 것이다. 이를 위해 문제를 `(left_x, left_y, right_x, right_y)` 네 개의 연속 좌표 값을 예측하는 supervised regression task로 정의한다.

### 2. 기존 연구: 최근 선행 연구들은 이 문제를 어떻게 다루는가?

기존 연구는 대체로 강력한 landmark detector를 쓰거나 전체 얼굴 de-identification을 수행한다. 얼굴 개인정보 보호 연구는 민감한 픽셀을 마스킹하거나 블러 처리하고, 더 최근에는 GAN/diffusion 기반 생성, representation-level privacy transformation, privacy-preserving image acquisition 등을 통해 얼굴 identity를 변환하거나 대체한다.

이 프로젝트의 차별점은 더 가벼운 eye-region anonymization prototype을 직접 구현한다는 점이다. RetinaFace와 같은 pretrained landmark detector는 정확하지만, 여기서는 작은 annotated dataset만으로 simple custom CNN이 투명한 self-built anonymization pipeline을 지원할 수 있는지도 함께 확인한다.

### 3. 핵심 도전 과제: 최근 연구들이 충분히 해결하지 못하지만, 우리의 방법이 해결하려는 문제는 무엇인가?

핵심 도전 과제는 small-data, interpretable-prototype 환경에서도 눈 위치를 안정적으로 찾는 것이다. 예측 좌표가 크게 흔들리면 black-box가 눈의 일부를 놓칠 수 있고, 그러면 개인정보 보호 장치로서 실패한다.

이 프로젝트는 대규모 pretrained detector에만 의존하지 않기 위해 eye-coordinate regression 문제를 따로 분리한다. Comparison image를 학습에서 제외한 뒤, compact model이 unseen BioID image에서도 privacy-preserving masking을 지원할 수 있는지 평가한다.

### 4. 우리의 방법: 3번 문제를 해결하기 위해 어떤 방법을 사용했는가?

우리의 방법은 custom CNN으로 눈 좌표를 예측한 뒤 black-box anonymization으로 연결하는 pipeline이다. BioID Face Database의 grayscale 얼굴 이미지와 왼쪽/오른쪽 눈 좌표가 담긴 `.eye` annotation file을 사용한다.

데이터와 모델은 전체 과정을 추적하기 쉽도록 단순하게 구성했다. 이미지는 one-channel `286 x 384` tensor로 전처리하고, label은 네 개의 floating-point pixel coordinate로 로드하며, EyeCNN 구조는 다음과 같다.

- Input: `(1, 286, 384)` shape의 grayscale face image.
- Feature extractor: `32`, `64`, `128` channel을 갖는 세 개의 `Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d` block.
- Regression head: flatten, dropout `0.3`, `Linear(128 * 35 * 48 -> 256)`, ReLU, `Linear(256 -> 4)`.
- Training: Adam optimizer, learning rate `1e-3`, MSE loss, `250` epochs.

평가는 custom CNN의 성능을 과대평가하지 않도록 설계했다. 21장의 comparison image를 학습 전에 먼저 분리하고, 나머지 1,500장의 이미지를 training/evaluation에 사용했으며, 예측 좌표는 margin이 포함된 black-box 영역으로 변환했다. RetinaFace는 main model이 아니라 strong external baseline으로 사용했다.

### 5. 실험 결과: 우리의 방법이 실제로 문제를 해결했음을 어떻게 보였는가?

실험 결과는 custom CNN이 pipeline 검증에는 유용하지만, 성능 자체는 RetinaFace가 더 좋다는 점을 보여준다. 근거는 완성된 end-to-end privacy pipeline에서 나온다.

- Training loss가 충분히 수렴하여 예측 좌표가 ground truth에 시각적으로 가까워졌다.
- Held-out comparison image에서 예측한 eye point가 annotation으로부터 작은 pixel error 범위 안에 있음을 확인했다.
- Black-box visualization을 통해 unseen image에서도 양쪽 눈이 안정적으로 덮이는지 확인했다.
- RetinaFace와의 비교를 통해 custom model과 강력한 pretrained detector 사이의 성능 차이를 명확히 보였다.

수치 결과는 RetinaFace가 모든 지표에서 custom CNN보다 우수하다는 점을 확인한다. 21장의 holdout image 기준 custom CNN은 MAE `2.84`, MSE `12.23`, Euclidean error `6.48`을 기록했고, RetinaFace는 MAE `1.46`, MSE `2.92`, Euclidean error `3.23`을 기록했다. 세 지표 모두 낮을수록 좋기 때문에, custom CNN은 최고 성능 모델이 아니라 pipeline 검증용 lightweight prototype이고 RetinaFace는 더 강한 baseline으로 해석하는 것이 맞다.

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
