#!/usr/bin/bash
mvn exec:java -D exec.mainClass=com.greekengineering.lecturetoc.App -Dexec.args="'$1' '$2'"
