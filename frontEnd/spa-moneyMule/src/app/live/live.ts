import { Component } from '@angular/core';
import { Results } from '../results';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-live',
  imports: [FormsModule, CommonModule],
  templateUrl: './live.html',
  styleUrl: './live.css'
})
export class Live {

  constructor(private api: Results) {}

  senderAccountId = '';
  receiverAccountId = '';
  amount: number | null = null;
  time = '';
  senderBank = '';
  receiverBank = '';
  model = '';
  result: any = null;

  submitModel(model: 'xgboost' | 'vae', data: any) {

    // this.result = null;

    const payload = {
      model,
      senderAccountId: this.senderAccountId,
      receiverAccountId: this.receiverAccountId,
      amount: this.amount,
      time: this.time,
      senderBank: this.senderBank,
      receiverBank: this.receiverBank
    };
    console.log(payload);
    this.api.fetchLive(payload).subscribe({
      next: (resp : any) => {
        console.log("transaction details sent");

        this.api.fetchFlag().subscribe((flag)=>{
          this.result = flag;
        })
      },
      error: (err : any) => {
        console.error(err);
        alert('Error processing transaction');
      },
    });
  }

}
