# AI Pronunciation Trainer - Guide for AI Agents

## Initial project structure

```plaintext
.
├── AIModels.py
├── ai-pronunciation-trainer.code-workspace
├── databases
│   ├── data_de.csv
│   └── data_en.csv
├── data_de_en_2.pickle
├── GEMINI.md
├── images
│   └── MainScreen.jpg
├── lambdaGetSample.py
├── lambdaSpeechToScore.py
├── lambdaTTS.py
├── LICENSE
├── ModelInterfaces.py
├── models.py
├── pronunciationTrainer.py
├── README.md
├── requirements.txt
├── RuleBasedModels.py
├── static
│   ├── ASR_bad.wav
│   ├── ASR_good.wav
│   ├── ASR_okay.wav
│   ├── css
│   │   └── style-new.css
│   └── javascript
│       └── callbacks.js
├── templates
│   └── main.html
├── unitTests.py
├── utilsFileIO.py
├── webApp.py
├── whisper_wrapper.py
├── WordMatching.py
└── WordMetrics.py
```

## Architecture and Main Components

This project is a web application for evaluating and improving pronunciation using AI. The main components are:

### Key Interfaces (ModelInterfaces.py)

- `IASRModel`: Interface for voice recognition models (ASR)
- `ITranslationModel`: Interface for translation
- `ITextToSpeechModel`: Interface for voice synthesis (TTS)

### Models and Services

- ASR: Uses Whisper as main model (`whisper_wrapper.py`)
- TTS: Uses Silero Models for voice synthesis (`models.py`)
- Pronunciation evaluation: Implemented in `pronunciationTrainer.py`

### Audio Processing

- Audio is processed at 16kHz (`transform = Resample(orig_freq=48000, new_freq=16000)`)
- ffmpeg is required for audio processing

## Workflow

1. Frontend (`templates/main.html` and `static/javascript/callbacks.js`):

   - Web interface that captures audio and displays results
   - Handles interaction with backend via REST API

2. Backend (Lambdas):
   - `lambdaGetSample.py`: Gets sample texts from databases
   - `lambdaSpeechToScore.py`: Processes and evaluates audio
   - `lambdaTTS.py`: Generates audio for examples

## Project Conventions

### Language Management

Languages are managed at various levels:

1. Database: CSV files in `databases/` (example: `data_en.csv`, `data_de.csv`)
2. ASR Models: Configured in `models.py:getASRModel()`
3. Frontend: Configured in `callbacks.js` and `main.html`

### Extensibility

To add a new language:

```python
# In models.py
def getASRModel(language: str):
    if language == 'new_language':
        # Configure ASR model
```

## Development and Testing

### Local Setup

```bash
pip install -r requirements.txt
# Install ffmpeg if not present
python webApp.py
```

### Testing

- Unit tests are in `unitTests.py`
- Use Chrome browser for frontend testing

## Integration Points

### External Dependencies

- Whisper: Main ASR engine
- Silero Models: TTS models via PyTorch Hub
- ffmpeg: Audio processing

### APIs and Communication

- Frontend → Backend: POST requests with base64 audio
- Endpoints defined in lambda functions

## Development Guidelines

### Documentation

- All documentation and inline comments MUST be in English
- Follow existing documentation patterns in the codebase
- Document any deviations from standard implementation patterns

## Planned Pedagogical Improvements

### Project Direction: A Fork for a New Vision

After careful consideration, the decision has been made to develop this project as a distinct fork rather than contributing a series of large pull requests to the original repository. The planned changes are substantial and architectural in nature, effectively creating a "Ship of Theseus" scenario where the final project would be fundamentally different from the original.

This approach allows for greater freedom to innovate and refactor towards a new vision focused on:

- **Simplified Maintenance and Deployment:** Replacing the current Flask frontend with a more modern and simpler framework like Streamlit or Gradio.
- **Enhanced Code Quality:** Enforcing SonarQube standards, implementing comprehensive type hinting, and improving inline documentation.
- **Pedagogical Extensibility:** Introducing a more flexible and user-friendly system for adding new languages and learning content via YAML configuration files.

This fork will, of course, fully respect the original work by:

- **Maintaining the AGPL v3 License.**
- **Providing clear attribution and thanks to the original author, Thiagohgl, in the README.md.**

### Development Roadmap

In order of priority and complexity:

1. **Code Cleanup and Refactoring:**
   - Remove unused dependencies (`sqlalchemy`, `pickle-mixin`).
   - Eliminate redundant code, such as the manual `pickle`-based model caching which is already handled by the `transformers` library.
   - Refactor modules to improve clarity and adherence to single-responsibility principles (e.g., simplifying `WordMatching.py`).

2. **Logging and Error Handling:**
   - Implement logging and comprehensive error handling with stack traces.

   - These modifications are essential to be able to work comfortably and safely on the rest of the changes.

3. **YAML-based Language Management:**
   - Modify the way new languages are added to make it accessible to people without advanced technical knowledge.

   - In a directory called "languages", there will be one YAML file for each language.
   - The filenames will consist of the language's ISO code and optionally an underscore (\_) and a regional variant code.
   - The content of each file will be:
     - A description explaining the variant's characteristics
     - A list of models to use for:
       - Automatic speech recognition
       - Audio generation
       - IPA phonetic transcription
       - Each model will contain:
         - A model name or identifier
         - The name of the source, repository, or method used to locate and download it
         - A description explaining the model's characteristics
         - An indication of whether it should be loaded on a GPU or CPU
   - Hints on how to implement it and which parts of the code need to be modified can be found in the README.md section "How do I add a new language?"
     - New database file
     - Language configuration in `models.py`
     - Frontend language selector update

4. **British English Support:**
   - Follow the language addition method described in README.md
   - Initially use American English phrases (copy of data_en.csv)
   - Treat as a new language in the system
   - Document any necessary deviations from standard language implementation

5. **Frontend Overhaul (Streamlit/Gradio):**

6. **Learning Mode Controls**
   - Add checkboxes to toggle visibility of:
     - Audio sample playback button
     - English text display
     - IPA transcription (hidden if either audio or text is hidden)
   - Constraints:
     - At least one of audio/text must remain visible
     - Purpose: Enable focused practice modes
       - Memory mode: text only, no IPA/audio
       - Listening mode: audio only, no text/IPA

7. **Structured Learning Sets:**
   - Reorganize language datasets into subdirectories
   - New structure (example):

   ```plaintext
   /datasets/
   ├── en/
   │    ├── basics.csv
   │    ├── wh_questions.csv
   │    ├── past_simple.csv
   │    ├── phoneme_th.csv
   │    └── lesson_3_vocab.csv
   [...]
   ```

   - Features:
     - Users can add as many lists as they want
     - Add dataset dropdown after language selection
     - Dataset titles from filename
     - Optional sequential progression through lists
   - Purpose: Enable focused practice on specific:
     - Phonetic patterns
     - Grammar structures
     - Vocabulary sets
     - Lesson-specific content

### Development Methodology

1. Develop changes in independent branches
2. Keep commits small and self-contained
3. Document modifications in English, both in code comments and in a changelog.
4. Validate functionality with basic tests for each change.

### Notes for Agents

- Prioritize solutions that don't alter core architecture
- Suggest basic tests for proposed changes
- Alert if a change might affect existing functionality
- Follow existing patterns for language implementation
- Keep all documentation in English

### Description of the development system

```plaintext
System:
  Kernel: 6.14.0-34-generic arch: x86_64 bits: 64 compiler: gcc v: 13.3.0 clocksource: tsc
  Desktop: Cinnamon v: 6.4.8 tk: GTK v: 3.24.41 wm: Muffin v: 6.4.1 dm: LightDM v: 1.30.0
    Distro: Linux Mint 22.2 Zara base: Ubuntu 24.04 noble
Machine:
  Type: Desktop System: HP product: HP Pavilion Gaming Desktop TG01-0xxx v: N/A
    Chassis: type: 3
  Mobo: HP model: 8643 v: SMVB  part-nu: 8PJ18EA#AB9
    uuid: a6279bb7-082d-039c-c27e-e7470e65aa3f UEFI: AMI v: F.47 date: 11/28/2023
CPU:
  Info: 6-core model: AMD Ryzen 5 3600 bits: 64 type: MT MCP smt: enabled arch: Zen 2 rev: 0 cache:
    L1: 384 KiB L2: 3 MiB L3: 32 MiB
  Speed (MHz): avg: 2550 high: 3600 min/max: 2200/4208 boost: enabled volts: 1.1 V
    ext-clock: 100 MHz cores: 1: 3600 2: 2200 3: 2200 4: 2200 5: 3600 6: 2200 7: 2200 8: 2200 9: 3600
    10: 2200 11: 2200 12: 2200 bogomips: 86232
  Flags: avx avx2 ht lm nx pae sse sse2 sse3 sse4_1 sse4_2 sse4a ssse3 svm
Graphics:
  Device-1: NVIDIA TU117 [GeForce GTX 1650] vendor: Hewlett-Packard driver: nvidia v: 580.95.05
    arch: Turing pcie: speed: 2.5 GT/s lanes: 16 ports: active: none off: HDMI-A-1 empty: DVI-D-1
    bus-ID: 0c:00.0 chip-ID: 10de:1f82 class-ID: 0300
  Display: server: X.Org v: 21.1.11 with: Xwayland v: 23.2.6 driver: X: loaded: nvidia
    unloaded: fbdev,modesetting,nouveau,vesa gpu: nv_platform,nvidia,nvidia-nvswitch display-ID: :0
    screens: 1
  Screen-1: 0 s-res: 1920x1080 s-dpi: 101 s-size: 483x272mm (19.02x10.71") s-diag: 554mm (21.82")
  Monitor-1: HDMI-A-1 mapped: HDMI-0 note: disabled model: Lenovo D22-10
    res: 1920x1080 hz: 60 dpi: 102 size: 476x268mm (18.74x10.55") diag: 546mm (21.5") modes:
    max: 1920x1080 min: 640x480
  API: EGL v: 1.5 hw: drv: nvidia nouveau drv: nvidia platforms: device: 0 drv: nvidia device: 1
    drv: nouveau device: 2 drv: swrast gbm: drv: nvidia surfaceless: drv: nvidia x11: drv: nvidia
    inactive: wayland
  API: OpenGL v: 4.6.0 compat-v: 4.5 vendor: nvidia mesa v: 580.95.05 glx-v: 1.4
    direct-render: yes renderer: NVIDIA GeForce GTX 1650/PCIe/SSE2
Audio:
  Device-1: NVIDIA vendor: Hewlett-Packard driver: snd_hda_intel v: kernel pcie: speed: 8 GT/s
    lanes: 16 bus-ID: 0c:00.1 chip-ID: 10de:10fa class-ID: 0403
  Device-2: AMD Starship/Matisse HD Audio vendor: Hewlett-Packard driver: snd_hda_intel v: kernel
    pcie: speed: 16 GT/s lanes: 16 bus-ID: 0e:00.4 chip-ID: 1022:1487 class-ID: 0403
  API: ALSA v: k6.14.0-34-generic status: kernel-api
  Server-1: PipeWire v: 1.0.5 status: n/a (root, process) with: 1: pipewire-pulse status: active
    2: wireplumber status: active 3: pipewire-alsa type: plugin
Network:
  Device-1: Realtek RTL8822CE 802.11ac PCIe Wireless Network Adapter vendor: Hewlett-Packard
    driver: rtw_8822ce v: N/A pcie: speed: 2.5 GT/s lanes: 1 port: e000 bus-ID: 0a:00.0
    chip-ID: 10ec:c822 class-ID: 0280
  IF: wlp10s0 state: down
  Device-2: Realtek RTL8111/8168/8211/8411 PCI Express Gigabit Ethernet
    vendor: Hewlett-Packard RTL8111/8168/8411 driver: r8169 v: kernel pcie: speed: 2.5 GT/s lanes: 1
    port: d000 bus-ID: 0b:00.0 chip-ID: 10ec:8168 class-ID: 0200
  IF: enp11s0 state: up speed: 100 Mbps duplex: full
  IF-ID-1: docker0 state: down
Bluetooth:
  Device-1: Realtek Bluetooth Radio driver: btusb v: 0.8 type: USB rev: 1.0 speed: 12 Mb/s lanes: 1
    bus-ID: 1-12:3 chip-ID: 0bda:b00c class-ID: e001
  Report: hciconfig ID: hci0 rfk-id: 0 state: up  bt-v: 5.1 lmp-v: 10
    sub-v: d2e3 hci-v: 10 rev: cc6 class-ID: 7c0104
Drives:
  Local Storage: total: 2.96 TiB lvm-free: 245.04 GiB used: 1.19 TiB (40.1%)
  ID-1: /dev/nvme0n1 vendor: SK Hynix model: BC501 HFM256GDJTNG-8310A size: 238.47 GiB
    speed: 15.8 Gb/s lanes: 2 tech: SSD  fw-rev: 80002C00 temp: 55.9 C scheme: GPT
  ID-2: /dev/sda vendor: Western Digital model: WD10SMZW-11Y0TS0 size: 931.48 GiB type: USB
    rev: 3.1 spd: 5 Gb/s lanes: 1 speed: <unknown> tech: HDD rpm: 5400  fw-rev: 1021
    scheme: GPT
  ID-3: /dev/sdb vendor: Western Digital model: WD20EZRZ-00Z5HB0 size: 1.82 TiB speed: 6.0 Gb/s
    tech: HDD rpm: 5400  fw-rev: 0A80 scheme: GPT
Partition:
  ID-1: / size: 78.68 GiB used: 18.83 GiB (23.9%) fs: ext4 dev: /dev/sdb3
  ID-2: /boot/efi size: 598.8 MiB used: 98.4 MiB (16.4%) fs: vfat dev: /dev/nvme0n1p1
  ID-3: /home size: 342.29 GiB used: 201.66 GiB (58.9%) fs: ext4 dev: /dev/dm-1 mapped: HD2T-HOME
  ID-4: /tmp size: 78.19 GiB used: 70 MiB (0.1%) fs: ext4 dev: /dev/dm-5 mapped: HD2T-TMP
Swap:
  ID-1: swap-1 type: partition size: 32.06 GiB used: 508 KiB (0.0%) priority: -2 dev: /dev/dm-6
    mapped: HD2T-SWAP
USB:
  Hub-1: 1-0:1 info: hi-speed hub with single TT ports: 14 rev: 2.0 speed: 480 Mb/s lanes: 1
    chip-ID: 1d6b:0002 class-ID: 0900
  Device-1: 1-11:2 info: Silicon Labs USB OPTICAL MOUSE type: mouse driver: hid-generic,usbhid
    interfaces: 1 rev: 2.0 speed: 1.5 Mb/s lanes: 1 power: 100mA chip-ID: 10c4:8108 class-ID: 0301
  Device-2: 1-12:3 info: Realtek Bluetooth Radio type: bluetooth driver: btusb interfaces: 2
    rev: 1.0 speed: 12 Mb/s lanes: 1 power: 500mA chip-ID: 0bda:b00c class-ID: e001
  Device-3: 1-14:4 info: Darfon USB Keyboard type: keyboard,HID driver: hid-generic,usbhid
    interfaces: 2 rev: 2.0 speed: 1.5 Mb/s lanes: 1 power: 100mA chip-ID: 0d62:d93f class-ID: 0300
  Hub-2: 2-0:1 info: super-speed hub ports: 8 rev: 3.1 speed: 10 Gb/s lanes: 1 chip-ID: 1d6b:0003
    class-ID: 0900
  Hub-3: 3-0:1 info: hi-speed hub with single TT ports: 4 rev: 2.0 speed: 480 Mb/s lanes: 1
    chip-ID: 1d6b:0002 class-ID: 0900
  Hub-4: 4-0:1 info: super-speed hub ports: 4 rev: 3.1 speed: 10 Gb/s lanes: 1 chip-ID: 1d6b:0003
    class-ID: 0900
  Device-1: 4-3:2 info: Western Digital Elements 25A2 type: mass storage driver: usb-storage
    interfaces: 1 rev: 3.1 speed: 5 Gb/s lanes: 1 power: 896mA chip-ID: 1058:25a2 class-ID: 0806

Sensors:
  System Temperatures: cpu: 66.5 C mobo: N/A gpu: nvidia temp: 34 C
  Fan Speeds (rpm): N/A gpu: nvidia fan: 30%
Repos:
  Packages: pm: dpkg pkgs: 2206
  No active apt repos in: /etc/apt/sources.list
  Active apt repos in: /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list
    1: deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https: //developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /
  Active apt repos in: /etc/apt/sources.list.d/docker.list
    1: deb [arch="amd64" signed-by=/etc/apt/keyrings/docker.gpg] https: //download.docker.com/linux/ubuntu noble stable
  Active apt repos in: /etc/apt/sources.list.d/google-chrome.list
    1: deb [arch=amd64] https: //dl.google.com/linux/chrome/deb/ stable main
  Active apt repos in: /etc/apt/sources.list.d/microsoft-edge.list
    1: deb [arch=amd64] https: //packages.microsoft.com/repos/edge/ stable main
  Active apt repos in: /etc/apt/sources.list.d/nodesource.list
    1: deb [arch=amd64 signed-by=/usr/share/keyrings/nodesource.gpg] https: //deb.nodesource.com/node_20.x nodistro main
  Active apt repos in: /etc/apt/sources.list.d/official-package-repositories.list
    1: deb http: //packages.linuxmint.com zara main upstream import backport
    2: deb http: //archive.ubuntu.com/ubuntu noble main restricted universe multiverse
    3: deb http: //archive.ubuntu.com/ubuntu noble-updates main restricted universe multiverse
    4: deb http: //archive.ubuntu.com/ubuntu noble-backports main restricted universe multiverse
    5: deb http: //security.ubuntu.com/ubuntu/ noble-security main restricted universe multiverse
  Active apt repos in: /etc/apt/sources.list.d/vscode.sources
    1: deb [arch=amd64] https: //packages.microsoft.com/repos/code stable main
Info:
  Memory: total: 32 GiB available: 31.25 GiB used: 4.84 GiB (15.5%)
  Processes: 422 Power: uptime: 3h 22m states: freeze,mem,disk suspend: deep wakeups: 0
    hibernate: platform Init: systemd v: 255 target: graphical (5) default: graphical
  Compilers: gcc: 13.3.0 Client: Unknown python3.12 client inxi: 3.3.34
```
