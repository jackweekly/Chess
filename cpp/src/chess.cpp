#include "chess.h"
#include <stdexcept>

ChessBoard::ChessBoard() {}
ChessBoard::ChessBoard(const std::string& /*fen_str*/) {
    // TODO: parse FEN
}

std::string ChessBoard::fen() const {
    // TODO: return proper FEN
    return "8/8/8/8/8/8/8/8 w - - 0 1";
}

bool ChessBoard::is_game_over() const {
    // TODO: real termination check
    return false;
}

std::string ChessBoard::result() const {
    // TODO: real result
    return "1/2-1/2";
}

std::vector<std::string> ChessBoard::legal_moves() const {
    // TODO: generate legal moves; placeholder empty to avoid crashes
    return {};
}

void ChessBoard::push(const std::string& /*uci*/) {
    // TODO: apply move
}
