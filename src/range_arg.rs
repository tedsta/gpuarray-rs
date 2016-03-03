use std::ops::{Range, RangeFrom, RangeTo, RangeFull};

/*pub struct RangeArg {
    pub start: usize,
    pub end: usize,
}

impl<T> Deref for T where T: IRangeArg {
    type Target = RangeArg;

    fn deref(&self) -> &RangeArg {
        &self.value
    }
}*/

////////////////////////////////////////////////////////////////////////////////////////////////////

pub trait RangeArg {
    fn start(&self) -> usize;
    fn end(&self) -> usize;
}

impl RangeArg for Range<usize> {
    fn start(&self) -> usize {
        self.start
    }

    fn end(&self) -> usize {
        self.end
    }
}

impl RangeArg for RangeFrom<usize> {
    fn start(&self) -> usize {
        self.start
    }

    fn end(&self) -> usize {
        0
    }
}

impl RangeArg for RangeTo<usize> {
    fn start(&self) -> usize {
        0
    }

    fn end(&self) -> usize {
        self.end
    }
}

impl RangeArg for RangeFull {
    fn start(&self) -> usize {
        0
    }

    fn end(&self) -> usize {
        0
    }
}

impl RangeArg for usize {
    fn start(&self) -> usize {
        *self
    }

    fn end(&self) -> usize {
        *self + 1
    }
}
