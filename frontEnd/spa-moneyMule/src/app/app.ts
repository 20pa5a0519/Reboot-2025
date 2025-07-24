import { Component } from '@angular/core';
import { RouterLink, RouterOutlet } from '@angular/router';
import { NgxGraphModule } from '@swimlane/ngx-graph';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, RouterLink, NgxGraphModule],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  protected title = 'spa-moneyMule';

  constructor() {}
}
