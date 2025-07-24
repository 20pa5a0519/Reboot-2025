import { TestBed } from '@angular/core/testing';

import { Results } from './results';

describe('Results', () => {
  let service: Results;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(Results);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
