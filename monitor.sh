#!/bin/bash
# This script monitors CPU and memory usage
#
rm log.csv
echo "CPU,MEM" > log.csv

while :
do 
  # Get the current usage of CPU and memory
  cpuUsage=$(mpstat 1 1 | awk '/AM/&&/all/ { print 100 - $NF }')
  memUsage=$(free -m | awk '/Mem/{ print $3 }')

  # Print the usage
  echo "Memory Usage: $memUsage MB"
  echo "CPU Usage: $cpuUsage"
  echo "---"
 
  echo "$cpuUsage,$memUsage" >> log.csv

  # Sleep for 1 second
  sleep 1
done
