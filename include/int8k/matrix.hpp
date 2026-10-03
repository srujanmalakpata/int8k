// Non-owning views over row-major float32 matrices.
#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>

namespace int8k {

/// True when `size == count * per_item`, checked without computing the (possibly overflowing)
/// product.
[[nodiscard]] constexpr bool size_matches(std::size_t size, std::size_t count,
                                          std::size_t per_item) {
  if (per_item == 0) return size == 0;
  return size % per_item == 0 && size / per_item == count;
}

/// A read-only, row-major view of a `rows x cols` float32 matrix.
///
/// The view does not own its memory; the caller keeps the buffer alive. The invariant
/// `data().size() == rows() * cols()` is checked once in the constructor (without overflowing
/// the product) and the fields are private, so it cannot be broken afterwards.
class MatrixView {
 public:
  MatrixView() = default;
  MatrixView(std::span<const float> d, std::size_t r, std::size_t c)
      : data_(d), rows_(r), cols_(c) {
    if (!size_matches(d.size(), r, c)) {
      throw std::invalid_argument("MatrixView: data.size() must equal rows * cols");
    }
  }

  [[nodiscard]] std::span<const float> data() const { return data_; }
  [[nodiscard]] std::size_t rows() const { return rows_; }
  [[nodiscard]] std::size_t cols() const { return cols_; }

  [[nodiscard]] std::span<const float> row(std::size_t r) const {
    return data_.subspan(r * cols_, cols_);
  }

 private:
  std::span<const float> data_;
  std::size_t rows_ = 0;
  std::size_t cols_ = 0;
};

}  // namespace int8k
