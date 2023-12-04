package org.example;

public class util {
    public static int[][] createboard(int[][] board){
        int[][] nb = new int[7][7];
        for(int i=0;i<board.length;i++){
            System.arraycopy(board[i], 0, nb[i], 0, board[i].length);
        }
        return nb;
    }
    public static int[] newloc(int[] playerloc){
        int[] nl = new int[2];
        System.arraycopy(playerloc, 0, nl, 0, playerloc.length);
        return nl;
    }
}
