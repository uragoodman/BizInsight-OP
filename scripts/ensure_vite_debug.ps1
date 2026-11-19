<##
Ensure the Vue development server is reachable before VS Code launches Chrome.

This script deliberately exits after Vite is ready. VS Code therefore treats it
as an ordinary pre-launch task instead of a never-ending watch task, avoiding
the misleading "doesn't have a problemMatcher" dialog.
##>
<# ============================================================
   调试辅助脚本：确保 Vite 开发服务器就绪后再启动 Chrome
   作用：VS Code 启动 Chrome 调试前调用本脚本，等待 Vite 在 web/ 下启动完毕。
   设计要点：脚本在 Vite 就绪后主动退出，VS Code 因此把它当作普通 pre-launch
   任务（而非永不结束的 watch 任务），避免出现误导性的 "doesn't have a
   problemMatcher" 弹窗。
   ============================================================ #>

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
# Vite 开发服务器端口（需与 web/vite.config.ts 的 server.port 一致）。
$port = 15299

# 检测 127.0.0.1:$port 是否有端口在监听；有则说明 Vite 已就绪。
function Test-ViteReady {
    return $null -ne (Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
}

# 若 Vite 已在运行则直接通过，避免重复拉起一个实例。
if (Test-ViteReady) {
    Write-Output "Vite is already listening on http://127.0.0.1:$port/"
    exit 0
}

# 后台以隐藏窗口启动 npm run dev（工作目录为 web/）。
$npm = (Get-Command npm.cmd -ErrorAction Stop).Source
Start-Process -FilePath $npm -ArgumentList @("run", "dev") -WorkingDirectory (Join-Path $projectRoot "web") -WindowStyle Hidden | Out-Null

# 主流程：在 15 秒内每 300ms 轮询一次端口，就绪即返回成功。
$deadline = (Get-Date).AddSeconds(15)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 300
    if (Test-ViteReady) {
        Write-Output "Vite is ready at http://127.0.0.1:$port/"
        exit 0
    }
}

# 超时仍未就绪：抛出错误，提示手动到 web/ 目录运行 npm run dev 排查输出。
throw "Vite did not start on http://127.0.0.1:$port/ within 15 seconds. Run 'npm run dev' in the web directory to inspect its output."
