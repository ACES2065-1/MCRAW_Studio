# MCRAW Studio — release verification

Each row is a published `MCRAWStudio.exe`. Use the SHA-256 to verify your
download came from this repository and hasn't been tampered with.

## How to verify

**Windows PowerShell**

```powershell
Get-FileHash -Algorithm SHA256 .\MCRAWStudio.exe
```

**Windows Command Prompt**

```cmd
certutil -hashfile MCRAWStudio.exe SHA256
```

The reported hash must match the row below for your version *exactly* (case
doesn't matter). If it doesn't, **don't run the file** — re-download from
the GitHub Release linked here, not from a third-party mirror.

## Releases

| Version | Date       | Size       | SHA-256                                                            | Release |
|---------|------------|------------|--------------------------------------------------------------------|---------|
| 0.1.0   | 2026-05-16 | 64,344,698 | `D40C54BEAB83E9C11CC8B89DC05429F8623F91130C329EB20EB22A59BE29C34E` | [v0.1.0](https://github.com/ACES2065-1/MCRAW_Studio/releases/tag/v0.1.0) |
| 0.1.1   | 2026-05-22 | 83,374,606 | `8078ACFDF31BAA46237DD382A62BE89EBF9C6A004C69A0D13AEBF90513844590` | [v0.1.1](https://github.com/ACES2065-1/MCRAW_Studio/releases/tag/v0.1.1) |
| 0.2.0   | 2026-05-22 | 83,432,454 | `E3CA92B638E094BD79645D36A27BD708B49331C36E9D6A3453A7E01F851D26C4` | [v0.2.0](https://github.com/ACES2065-1/MCRAW_Studio/releases/tag/v0.2.0) |
| 0.2.1   | 2026-05-22 | 83,432,622 | `C74E5277A91D45501850A591FB31B02FD9F93E6AC3997FB0B2EBA88A171B02B1` | [v0.2.1](https://github.com/ACES2065-1/MCRAW_Studio/releases/tag/v0.2.1) |

<!--
When cutting a new release:
  1. Build the .exe (PyInstaller).
  2. Get-FileHash -Algorithm SHA256 .\dist\MCRAWStudio.exe
  3. Tag a release on GitHub (e.g. v0.2.0) and attach BOTH:
       - MCRAWStudio.exe
       - MCRAWStudio.exe.sha256   (a one-line text file with the hash)
  4. Add a row to the table above with the date, size, hash, and release link.
  5. Commit + push.
-->
