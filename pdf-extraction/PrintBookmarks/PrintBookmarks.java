import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.interactive.documentnavigation.outline.PDDocumentOutline;
import org.apache.pdfbox.pdmodel.interactive.documentnavigation.outline.PDOutlineItem;
import org.apache.pdfbox.pdmodel.interactive.documentnavigation.outline.PDOutlineNode;
import java.io.File;

public class PrintBookmarks
{
    /**
     * This will print the documents data.
     *
     * @param args The command line arguments.
     *
     * @throws Exception If there is an error parsing the document.
     */
    public static void main( String[] args ) throws Exception
    {
        if( args.length != 1 )
        {
            System.exit(1);
        }

        PDDocument doc = PDDocument.load(new File(args[0]));
        PDDocumentOutline root = doc.getDocumentCatalog().getDocumentOutline();
        printBookmark(root, "");
    }

    /**
     * This will print the documents bookmarks to System.out.
     *
     * @param bookmark The bookmark to print out.
     * @param indentation A pretty printing parameter
     *
     * @throws IOException If there is an error getting the page count.
     */
    public static void printBookmark( PDOutlineNode bookmark, String indentation )
    {
        PDOutlineItem current = bookmark.getFirstChild();
        while( current != null )
        {
            System.out.println( indentation + current.getTitle() );
            printBookmark( current, indentation + "    " );
            current = current.getNextSibling();
        }

    }
}
