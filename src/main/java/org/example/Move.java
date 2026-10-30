package org.example;

import java.util.Objects;

public class Move {
    public int[] tryMove(int dx, int dy, int[] location, int[][] board, int[] enemyloc) {
        return tryMove(dx, dy, location, board, new int[][]{enemyloc});
    }

    /** others: 움직이는 말을 뺀 모든 말(상대 말과 내 다른 말)의 {row, col}. 이 칸으로는 걸어갈 수 없다. */
    public int[] tryMove(int dx, int dy, int[] location, int[][] board, int[][] others) {
        if (!canMove(location, dx, dy)) {
            return null;
        }
        for (int[] other : others) {
            if (!isthereplayer(location, dx, dy, other)) {
                return null;
            }
        }
        return move(dx, dy, location[0], location[1], board);
    }

    private boolean isthereplayer(int[] loc, int dx, int dy, int[] enemyloc){
        int row = loc[0];
        int col = loc[1];
        if (dy == 0) {
            return col + dx != enemyloc[1] || row != enemyloc[0];
        } else if (dx == 0) {
            return col != enemyloc[1] || row + dy != enemyloc[0];
        }
        return true;
    }


    private boolean isWithinBounds(int value) {
        return value >= 0 && value <= 6;
    }

    public boolean canMove(int[] loc, int dx, int dy) {
        int row = loc[0];
        int col = loc[1];

        if (dy == 0) {
            return isWithinBounds(col + dx);
        } else if (dx == 0) {
            return isWithinBounds(row + dy);
        }
        return false;
    }

    private int[] move(int dx, int dy, int row, int col, int[][] list) {
        if (dy == 0) {
            return movePiece(row, col, row, col + dx, 0, dx, list);
        } else if (dx == 0) {
            return movePiece(row, col, row + dy, col, dy, 0, list);
        }
        return null;
    }

    private int[] movePiece(int fromRow, int fromCol, int toRow, int toCol, int rowMove, int colMove, int[][] list) {
        if (list[toRow][toCol] == 0) {
            int hatchMove = 0;
            if (rowMove == 0) {
                while (canMove(new int[]{fromRow, fromCol + hatchMove}, colMove, 0)) {
                    if (list[fromRow][fromCol + hatchMove + colMove] != 1) {
                        hatchMove += colMove;
                    } else {
                        break;
                    }
                }
            } else if (colMove == 0) {
                while (canMove(new int[]{fromRow + hatchMove, fromCol}, 0, rowMove)) {
                    if (list[fromRow + hatchMove + rowMove][fromCol] != 1) {
                        hatchMove += rowMove;
                    } else {
                        break;
                    }
                }
            }

            if (rowMove == 0) {
                return new int[]{fromRow, fromCol + hatchMove, 1};
            }
            if (colMove == 0) {
                return new int[]{fromRow + hatchMove, fromCol, 1};
            }
            return null;
        }
        else {
            return new int[]{toRow, toCol, 0};
        }
    }
}
