#!/usr/bin/bash

###################################################################
# Wrapper that runs a process exclusively, as long as other process
# also use this wrapper.
#
# author: Daniel Campora (dcampora@cern.ch)
# date: 06/2021
###################################################################

MAX_ATTEMPTS=10
TIMEOUT=2
LOCK_FILE="/tmp/${USER}-lock"
PROCESS_HAS_RUN=0
ATTEMPTS=0

if [[ ! -f "${LOCK_FILE}" ]]
then
  touch ${LOCK_FILE}
  chmod 777 ${LOCK_FILE}
fi

# Create a file descriptor over the given lockfile.
exec {FD}<>$LOCK_FILE

while [ "${PROCESS_HAS_RUN}" -eq 0 ] && [ "${ATTEMPTS}" -lt "${MAX_ATTEMPTS}" ]
do
  if [ "${ATTEMPTS}" -gt 1 ]
  then
    echo Attempt $((${ATTEMPTS} + 1))...
  fi

  if ! flock -x -w ${TIMEOUT} ${FD}
  then
    ATTEMPTS=$((${ATTEMPTS} + 1))
  else
    # Run command
    $@

    # Mark it as run
    PROCESS_HAS_RUN=1
  fi
done

if [ "${PROCESS_HAS_RUN}" -eq 0 ]
then
  echo Process could not run after ${MAX_ATTEMPTS} attempts.
  echo Please try again or contact your labmate.
fi
