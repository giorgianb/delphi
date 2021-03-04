class StringStripper {
    static String prevString = "I am some text. I start with the next portion.";
    static String nextString = "I start with the next portion. I am some text.";

    public static void main(String args[]) {
        System.out.println(stripTrailingText(prevString, nextString));
    }

    public static String stripTrailingText(String text, String nextText) {
        int stop = 0;
        for (int i = 0; i < nextText.length(); ++i)
            if (text.endsWith(nextText.substring(0, i)))
                stop = i;

        if (stop != 0)
            return text.substring(0, text.length() - stop);
        else
            return text;
    }

}
