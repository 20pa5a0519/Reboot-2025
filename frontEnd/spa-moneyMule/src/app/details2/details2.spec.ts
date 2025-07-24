import { ComponentFixture, TestBed } from '@angular/core/testing';

import { Details2 } from './details2';

describe('Details2', () => {
  let component: Details2;
  let fixture: ComponentFixture<Details2>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [Details2]
    })
    .compileComponents();

    fixture = TestBed.createComponent(Details2);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
