#[derive(Debug, Clone)]
pub struct IntegerCompositions {
    k: usize,
    current: Vec<u64>,
    first: bool,
}

impl IntegerCompositions {
    pub fn new(n: usize, k: usize) -> Self {
        let mut current = vec![0; k];
        current[0] = n as u64;
        IntegerCompositions {
            k,
            current,
            first: true,
        }
    }
}

impl Iterator for IntegerCompositions {
    type Item = Vec<u64>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.first {
            self.first = false;
            return Some(self.current.clone());
        }

        for i in (0..self.k - 1).rev() {
            if self.current[i] > 0 {
                self.current[i] -= 1;
                let mut sum = 0;
                for j in i + 1..self.k {
                    sum += self.current[j];
                    self.current[j] = 0;
                }
                self.current[i + 1] = sum + 1;
                return Some(self.current.clone());
            }
        }

        None
    }
}
