#!/usr/bin/bash
CLASSPATH=pdfbox2.jar:./:/usr/share/java/commons-logging.jar:fontbox2.jar
java -cp $CLASSPATH GetBookmarks "$1"
