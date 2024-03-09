package org.example;

import java.util.Objects;

public class Move {
    public int[] tryMove(int dx, int dy, int[] location, int[][] board, int[] enemyloc) {
        if (canMove(location, dx, dy) && isthereplayer(location, dx, dy, enemyloc)) {
            return move(dx, dy, location[0], location[1], board);
        }
        return null;
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
            return movePiece(row, col, dy, col, dy, 0, list);
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
