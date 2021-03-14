package com.greekengineering.lecturetoc;

import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.interactive.documentnavigation.outline.PDDocumentOutline;
import org.apache.pdfbox.pdmodel.interactive.documentnavigation.outline.PDOutlineItem;
import org.apache.pdfbox.pdmodel.interactive.documentnavigation.outline.PDOutlineNode;
import java.io.File;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Map;
import org.apache.pdfbox.cos.COSName;
import org.apache.pdfbox.cos.COSBase;
import org.apache.pdfbox.cos.COSDictionary;
import org.apache.pdfbox.text.PDFTextStripper;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

public class App
{
    public static boolean SORT_BY_BEADS = false;
    public static boolean SORT_BY_POSITION = true;
    public static void main(String[] args) throws Exception
    {
        if (args.length != 2)
        {
            System.out.println("Need two files.");
            System.exit(1);
        }
        final String pdfPath = args[0];
        final String csvPath = args[1];

        PDDocument doc = PDDocument.load(new File(pdfPath));
        PDDocumentOutline root = doc.getDocumentCatalog().getDocumentOutline();
        ArrayList<PDOutlineItem> bookmarks = new ArrayList<>();
        getBookmarks(root, bookmarks);
        PDFTextStripper strip = new PDFTextStripper();
        ArrayList<String> texts = new ArrayList<>();
        strip.setShouldSeparateByBeads(SORT_BY_BEADS);
        strip.setSortByPosition(SORT_BY_POSITION);
        for (int i = 0; i < bookmarks.size() - 1; ++i) {
            strip.setStartBookmark(bookmarks.get(i));
            strip.setEndBookmark(bookmarks.get(i + 1));
            try {
                texts.add(strip.getText(doc).trim().replaceAll("\\s\\s*", " "));
                        } catch (IOException e) {
                            texts.add("");
                            System.out.println("Skipping " + bookmarks.get(i).getTitle());
                        }
        }
        strip.setStartBookmark(bookmarks.get(bookmarks.size() - 1));
        strip.setEndBookmark(null);
        texts.add(strip.getText(doc).trim().replaceAll("\\s\\s*", " "));

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(csvPath))) {
            CSVPrinter printer = new CSVPrinter(writer, CSVFormat.DEFAULT);
            for (int i = 0; i < bookmarks.size() && i < texts.size(); ++i) {
                final String title = bookmarks.get(i).getTitle();
                final String text = texts.get(i);
                printer.printRecord(title, text);
            }
            printer.flush();
        }
    }

    public static void getBookmarks(PDOutlineNode bookmark, ArrayList<PDOutlineItem> bookmarks)
    {
        PDOutlineItem current = bookmark.getFirstChild();
        while (current != null)
        {
            bookmarks.add(current);
            getBookmarks(current, bookmarks);
            current = current.getNextSibling();
        }

    }
}
