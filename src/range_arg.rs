use std::ops::{Range, RangeFrom, RangeTo, RangeFull};

/*pub struct RangeArg {
    pub start: Option<usize>,
    pub end: Option<usize>,
}

impl<T> Deref for T where T: IRangeArg {
    type Target = RangeArg;

    fn deref(&self) -> &RangeArg {
        &self.value
    }
}*/

////////////////////////////////////////////////////////////////////////////////////////////////////

pub trait RangeArg {
    fn start(&self) -> Option<usize>;
    fn end(&self) -> Option<usize>;
}

impl RangeArg for Range<usize> {
    fn start(&self) -> Option<usize> {
        Some(self.start)
    }

    fn end(&self) -> Option<usize> {
        Some(self.end)
    }
}

impl RangeArg for RangeFrom<usize> {
    fn start(&self) -> Option<usize> {
        Some(self.start)
    }

    fn end(&self) -> Option<usize> {
        None
    }
}

impl RangeArg for RangeTo<usize> {
    fn start(&self) -> Option<usize> {
        None
    }

    fn end(&self) -> Option<usize> {
        Some(self.end)
    }
}

impl RangeArg for RangeFull {
    fn start(&self) -> Option<usize> {
        None
    }

    fn end(&self) -> Option<usize> {
        None
    }
}

impl RangeArg for usize {
    fn start(&self) -> Option<usize> {
        Some(*self)
    }

    fn end(&self) -> Option<usize> {
        Some(*self + 1)
    }
}
