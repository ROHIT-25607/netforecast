# SIH 2026 Submission Guide

Checklist before sharing this repository link, following the standard
[NSUT-SIH-DEMO](https://github.com/NSUT-SIH-26/NSUT-SIH-DEMO) template.

## Required repository content

- [x] Actual source code is present (`src/`, `demo/app.py`).
- [x] `README.md` explains the project clearly.
- [x] PS ID and PS title are included (SIH 26153, NTRO).
- [x] Problem statement and proposed solution are explained.
- [x] Key features are listed.
- [x] Technology stack is listed.
- [x] Setup and run instructions work.
- [ ] Team members and roles are mentioned.
- [x] Important screenshots are added to `assets/screenshots/`.
- [x] Final PPT/presentation is placed in `submission/` (or an external
      link is added to `submission/PRESENTATION.md`).
- [x] Demo video link is added to `submission/DEMO.md` (optional).
- [ ] Repository is accessible to reviewers.

## Recommended structure

```text
netforecast/
├── README.md
├── SUBMISSION_GUIDE.md
├── submission/
│   ├── PRESENTATION.md
│   └── DEMO.md
├── src/
├── docs/
│   └── architecture.md
├── assets/
│   └── screenshots/
├── demo/
├── data/
├── models/ · models_real/
├── reports/
└── requirements.txt
```

## Presentation

Upload the final PPT/PPTX to `submission/` when the file size is suitable
for GitHub, using a clear filename such as `NetForecast_SIH2026_Presentation.pptx`.
If it is too large, use Google Drive/OneDrive and put the shareable viewer
link in `submission/PRESENTATION.md`.

## Demo video

Optional. If recorded (see `submission/DEMO.md` for the script), add its
YouTube/Google Drive link there and make sure it is accessible without
requesting permission.

## Screenshots

Put the most useful screens/results in `assets/screenshots/` — see that
folder's `README.md` for the recommended list and naming convention.

## Do not upload

- Passwords, API keys, access tokens
- `.env` files containing secrets
- Private credentials or other confidential information

## Before submission

Open the repository in a private/incognito browser window (or while
logged out) and verify a reviewer can access the code, PPT, screenshots,
documentation, and any submitted links.
