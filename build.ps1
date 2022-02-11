$env:CGO_ENABLED=1
$env:CGO_CFLAGS="-ID:\tools\amd-ocl\include"
$env:CGO_LDFLAGS="-LD:\tools\amd-ocl\lib\x86_64"

go build