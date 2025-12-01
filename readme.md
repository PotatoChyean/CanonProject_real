## PyTorch DLL 로드 오류 해결 보고서 (WinError 1114)

PyTorch 기반 FastAPI 서버 실행 중 발생했던 `OSError: [WinError 1114] DLL 초기화 루틴을 실행할 수 없습니다. Error loading "C:\...\torch\lib\c10.dll" or one of its dependencies` 오류에 대한 문제 진단 및 해결 과정을 보고서 형식으로 정리했습니다.

-----

### 1\. 📌 문제 개요

| 구분 | 내용 |
| :--- | :--- |
| **발생 오류** | `OSError: [WinError 1114] DLL 초기화 루틴을 실행할 수 없습니다.` |
| **원인** | PyTorch의 핵심 라이브러리인 `c10.dll`이 로드될 때, 해당 DLL이 의존하는 **시스템 종속성(Visual C++ 런타임 등)** 파일을 찾지 못하거나, **잘못된 순서로 로드**되어 초기화에 실패함. |
| **근본 원인** | 가상 환경의 내부 DLL 경로가 Windows의 **시스템/사용자 환경 변수(`Path`)에 수동으로 등록**되어 PyTorch의 정상적인 로드 메커니즘을 방해함. |
| **영향** | FastAPI 서버(`main.py`) 실행 불가. |

-----

### 2\. 🔍 초기 진단 및 해결 시도

| 단계 | 시도 내용 | 결과 | 비고 |
| :--- | :--- | :--- | :--- |
| **1단계** | **Visual C++ Redistributable 설치/복구** | 실패 | DLL 종속성 파일 자체의 누락 문제는 아니었음을 확인. |
| **2단계** | PyTorch 및 관련 패키지 **클린 재설치** | 실패 | `torchvision` 버전 태그 오류 (`+cpu`) 등 설치 자체에 어려움 발생. |
| **3단계** | **`CUDA_VISIBLE_DEVICES=""` 환경 변수 설정** | 실패 | 오류 원인이 GPU/CUDA 로드 시도 때문만은 아님을 확인. |
| **4단계** | PyTorch 버전 **다운그레이드 시도** | 실패 | 설치 인덱스 문제로 `1.x` 버전 설치 불가. |

-----

### 3\. 🎯 최종 진단 및 문제 해결 (핵심)

최종 진단 결과, 오류는 Windows의 **DLL 검색 순서 문제**였으며, 다음 조치를 통해 해결되었습니다.

#### A. 환경 변수 경로 정리 (핵심 해결)

1.  **진단:** `Path` 환경 변수에 **가상 환경 내부의 DLL 경로**가 등록되어 있음을 확인했습니다.
      * **문제 경로:** `C:\GIthub\CanonProject_real\.venv\Lib\site-packages\torch\lib`
2.  **조치:** Windows **시스템/사용자 환경 변수(`Path`)** 목록에서 위 경로를 **즉시 삭제**했습니다.
3.  **효과:** PyTorch의 DLL 로드 시퀀스에 대한 외부 간섭이 제거되었고, `import torch` 시 가상 환경 내부의 정상적인 로드 절차가 작동하게 되었습니다.

#### B. PyTorch 클린 설치 재시도 및 버전 통일

환경 변수 정리 후, 호환성 확보를 위해 PyTorch를 올바른 명령어로 재설치했습니다.

1.  **기존 환경 정리:**
    ```bash
    deactivate
    pip uninstall torch torchvision torchaudio -y
    ```
2.  **호환 버전 설치:** (Python 3.11 환경에 맞는 2.x 버전대 중 안정적인 버전 선택)
    ```bash
    pip install torch==2.3.1+cpu torchvision torchaudio -f https://download.pytorch.org/whl/cpu/torch_stable.html
    ```

#### C. 서버 실행

1.  **경로 이동:**
    ```bash
    cd server
    ```
2.  **GPU 충돌 방지 환경 변수 설정:** (안전 확보 차원에서 항상 실행)
    ```bash
    set CUDA_VISIBLE_DEVICES=""
    ```
3.  **서버 실행:**
    ```bash
    python main.py
    ```

-----

### 4\. 📝 향후 참고 사항

1.  **DLL Path 관리:** **어떤 경우에도 가상 환경(`venv`) 내부의 경로는 Windows 시스템/사용자 환경 변수(`Path`)에 직접 등록하지 않습니다.** 가상 환경 활성화가 그 역할을 대신합니다.
2.  **PyTorch 설치:** `pip install` 시 `+cpu` 태그가 인식되지 않을 경우, 반드시 `--index-url https://download.pytorch.org/whl/cpu` 옵션을 사용하거나, `-f https://download.pytorch.org/whl/cpu/torch_stable.html` 옵션을 사용해야 합니다.
3.  **WinError 1114 발생 시 최우선 조치:**
      * Visual C++ Redistributable 복구/재설치
      * **환경 변수(`Path`)에서 가상 환경 관련 DLL 경로 즉시 제거**
