@echo off
title Automated Docker Run

::format - 04-04-2022_24:00
: convention ('DDMMYYYY_HHMM')
: Sets the proper date and time stamp with 24Hr Time
echo docker desktop is running, do not press key. Please ignore below message.
timeout /t 180

::docker rm $(docker ps -a -f status=exited)
::echo old docker containers with status=exited, are deleted

set datetimef=%date:~-7,2%-%date:~-10,2%-%date:~-4,4%__%time:~0,2%_%time:~3,2%

echo %datetimef%

echo %datetimef% > "C:\CameraTesting\autocameratest2\data\docker.txt"
echo Docker.txt file has been created.

echo Docker run is executing
docker run -v C:\CameraTesting\autocameratest2\data:/autocameratest2/data -p 5000:5000  arunkg99/python-docker:3.2

pause



