import scipy.sparse as sparse 

def sparse_periodic_tridiag(
    n: int,
    main_val: float,
    sub_val: float,
    super_val: float
) -> sparse.csc_array:
    """Creates a sparse tridiagonal matrix, with the top right corner
    set to the same value as the first subdiagonal, and the bottom
    left corner set to the same value as the first superdiagonal.
    
    Args:
        n (int): The number of rows and columns of the desired array
            (should be >= 3)
        main_val (float): The value on the main diagonal
        sub_val (float): The value on the first subdiagonal and in
            the top right corner 
        super_val (float): The value on the first superdiagonal and
            in the bottom left corner

    Returns:
        sparse.csc_array: The resulting periodic tridiagonal array
            as a csc array
    """
    # Check the matrix is big enough
    if n < 3:
        raise ValueError("n must be >= 3")
    
    # Construct the main tridiagonal part
    T = sparse.dia_array((n,n))
    T.setdiag(main_val, 0)
    T.setdiag(sub_val, -1)
    T.setdiag(super_val, 1)

    # Fill in top right and bottom left corners
    T.setdiag(sub_val, n-1)
    T.setdiag(super_val, -(n-1))

    # Return this array as a csc array 
    return T.tocsc()


def sparse_block_row(
    shape: tuple[int, int],
    blocks: list[sparse.sparray | int]
) -> sparse.csc_array:
    """Create a block row sparse array of a given shape from given
    blocks.

    The entries in blocks may be either sparse arrays with the same
    number of rows as shape[0], or integers representing the number
    of columns to fill with zeros (where the number of rows of zeros
    is always equal to shape[0])

    The total number of columns should add up to shape[1].

    Args:
        shape (tuple[int, int]): The desired shape of the resulting
            row filled with these blocks
        blocks (list[sparse.sparray | int]): The blocks to fill this
            row with. May be either sparse arrays with the same
            number of rows as shape[0], or integers representing the
            number of columns to fill with zeros (where the number of
            rows of zeros is always equal to shape[0])

    Returns:
        sparse.csc_array: The resulting block row

    Raises:
        ValueError: If the resulting shape of combining the block
            arrays does not match the given shape
        ValueError: If a value in blocks is not a sparse array 
            or an integer
    """
    # Parse shape 
    rows, cols = shape 

    # Fill out complete list of block array entries
    total_cols_used = 0
    fleshed_out_blocks : list[sparse.sparray]= []
    for i, block in enumerate(blocks):
        # Parse block info
        if isinstance(block, int):
            block_cols = block      # The block represents the number of zero columns
            block = sparse.csc_array((rows, block_cols))
        elif isinstance(block, sparse.sparray):
            # Validate number of rows in given sparse matrix is equal
            # to number of rows in given shape
            block_rows, block_cols = block.shape
            if block_rows != rows:
                raise ValueError(f"Block in index {i} has {block_rows} rows; needs {rows} rows to be compatible")
        else:
            raise ValueError(f"Block in index {i} is of type {type(block)}; needs to be either sparse.sparray or int")
        
        # Validate number of columns is not too many for number of
        # columns in given shape
        total_cols_used += block_cols
        
        # If all is well, add fleshed-out block to the row
        fleshed_out_blocks.append(block)

    # Check we have the correct total number of columns 
    if total_cols_used != cols:
        raise ValueError(f"Total number of columns is {total_cols_used}; should be {cols}")
    
    # If all is well, create and return the block row as a
    # sparse CSC array
    return sparse.hstack(fleshed_out_blocks, format='csc')


def sparse_flip_ud(A: sparse.csc_array) -> sparse.csc_array:
    """Reverse the elements of a sparse array along axis 0 (up/down).
    
    Args:
        A (sparse.csc_array): A sparse array in CSC format.

    Returns:
        sparse.csc_array: The flipped version of A along axis 0

    Raises:
        ValueError: If A is not a CSC array, or cannot be converted
            to a CSC array
    """
    # Validate A is able to be a csc_array
    if not isinstance(A, sparse.csc_array):
        try:
            A = A.tocsc()
        except:
            raise ValueError("Sparse array A could not be converted to CSC format. Aborting sparse_flip_ud()")
    
    # Flip along axis 0
    B = sparse.csc_array(A.shape)
    B.data = A.data
    B.indices = -A.indices + A.shape[0] - 1
    B.indptr = A.indptr
    return B


def sparse_block_antidiag(
    blocks: list[sparse.csc_array]
) -> sparse.csc_array:
    """Creates a block antidiagonal array from a list of sparse
    arrays.
    
    Args:
        blocks (list[sparse.sparray]): The blocks to put on the 
            antidiagonal

    Returns:
        sparse.csc_array: The block-antidiagonal array formed from
            these blocks
    """
    # Invert blocks up/down
    inverted_blocks = [sparse_flip_ud(block) for block in blocks]

    # Put these into a block diagonal
    intermediate_block_diag = sparse.block_diag(inverted_blocks, format='csc')

    # Finally, flip this intermediate block diagonal matrix up/down
    # This moves it to a block antidiagonal matrix with the blocks
    # back in their rightful orientation
    return sparse_flip_ud(intermediate_block_diag)



if __name__ == "__main__":
    T = sparse_periodic_tridiag(5, 3.5, 1.5, 2.5)
    T2 = sparse_periodic_tridiag(4, 7.2, 4.2, 6.2)
    D_int = sparse.block_diag([T, T2], format='csc')
    result = sparse_flip_ud(D_int)

    

