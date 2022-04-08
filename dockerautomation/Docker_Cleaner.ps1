#docker rm $(docker ps -a -f status=exited -q)

#docker ps -a -q | % { docker rm $_ } 
Write-Output "Start Docker Container Cleaning"
docker ps -a -f status=exited -q | % { docker rm $_ } 
Write-Output "Finish Docker Container Cleaning"