import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.interactive.documentnavigation.outline.PDDocumentOutline;
import org.apache.pdfbox.pdmodel.interactive.documentnavigation.outline.PDOutlineItem;
import org.apache.pdfbox.pdmodel.interactive.documentnavigation.outline.PDOutlineNode;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Map;
import org.apache.pdfbox.cos.COSName;
import org.apache.pdfbox.cos.COSBase;
import org.apache.pdfbox.cos.COSDictionary;
import org.apache.pdfbox.text.PDFTextStripper;

public class GetBookmarks
{
    /**
     * This will print the documents data.
     *
     * @param args The command line arguments.
     *
     * @throws Exception If there is an error parsing the document.
     */
    public static void main(String[] args) throws Exception
    {
        if (args.length != 1)
        {
            System.exit(1);
        }

        PDDocument doc = PDDocument.load(new File(args[0]));
        PDDocumentOutline root = doc.getDocumentCatalog().getDocumentOutline();
        ArrayList<PDOutlineItem> bookmarks = new ArrayList<>();
        getBookmarks(root, bookmarks);
        PDFTextStripper strip = new PDFTextStripper();
        ArrayList<String> texts = new ArrayList<>();
//        strip.setShouldSeparateByBeads(true);
        strip.setSortByPosition(true);
//        PDOutlineItem testBookmark = bookmarks.get(10);
//        COSDictionary dict = testBookmark.getCOSObject();
//        for (Map.Entry<COSName, COSBase> cur: dict.entrySet())
//            System.out.println(cur.getKey().getName());
//        for (Map.Entry<COSName, COSBase> cur: dict.entrySet())
//            System.out.println(cur.getKey().getName());
//
//        System.exit(0);
        for (int i = 0; i < bookmarks.size() - 1; ++i) {
            strip.setStartBookmark(bookmarks.get(i));
            strip.setEndBookmark(bookmarks.get(i + 1));
            try {
                texts.add(strip.getText(doc).trim().replaceAll("\\s\\s*", " ").replaceAll(",", "_GREEK_ENGINEERING_COMMA_INTERNAL_USAGE_"));
            } catch (IOException e) {
                texts.add("");
                System.out.println("Skipping " + bookmarks.get(i).getTitle());
            }
        }
        strip.setStartBookmark(bookmarks.get(bookmarks.size() - 1));
        strip.setEndBookmark(null);
        texts.add(strip.getText(doc).trim().replaceAll("\\s\\s*", " "));

        try (PrintWriter writer = new PrintWriter(new FileWriter("chapters.csv"))) {
            for (int i = 0; i < bookmarks.size() && i < texts.size(); ++i) {
                final String title = bookmarks.get(i).getTitle().replace("\"", "\\\"");
                final String text = texts.get(i).replace("\"", "\\\"");
                writer.printf("\"%s\",\"%s\"\n", title, text);
            }
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
