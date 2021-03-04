#!/usr/bin/bash
CLASSPATH=pdfbox2.jar:./:/usr/share/java/commons-logging.jar:fontbox2.jar
javac -cp $CLASSPATH GetBookmarks.java
